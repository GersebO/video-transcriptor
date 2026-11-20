"""Separación de hablantes con sherpa-onnx (modelos ONNX en CPU, sin PyTorch ni token)."""

import bisect
import os
import tarfile
import urllib.request
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .config import Settings, get_settings
from .models import Segment, Transcript

_RELEASES = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
SEGMENTATION_ARCHIVE = f"{_RELEASES}/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
SEGMENTATION_FILE = "sherpa-onnx-pyannote-segmentation-3-0/model.onnx"
EMBEDDING_URL = (
    f"{_RELEASES}/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"
)
EMBEDDING_FILE = "3dspeaker_eres2net_base.onnx"

MIN_TURN_SECONDS = 0.5  # turnos más cortos se funden con el vecino (evita saltos falsos de hablante)
CLUSTER_THRESHOLD = 0.8  # con 0,5 una misma voz se partía en 9+ "personas" en una prueba real
MINOR_SPEAKER_SHARE = 0.05  # un "hablante" con menos de este % del habla (o de MINOR_SPEAKER_SECONDS)
MINOR_SPEAKER_SECONDS = 5.0  # es ruido de agrupado y se absorbe en el hablante principal más cercano
READ_BLOCK = 60 * 16000  # el WAV se lee por bloques de 1 min para no duplicar la memoria

ProgressCallback = Callable[[float, float], None]  # (avance, total)


@dataclass
class Turn:
    start: float
    end: float
    speaker_id: str  # "Persona 1", "Persona 2"... en orden de aparición


def diarization_dir(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).models_dir / "diarization"


def model_paths(settings: Settings | None = None) -> tuple[Path, Path]:
    d = diarization_dir(settings)
    return d / SEGMENTATION_FILE, d / EMBEDDING_FILE


def models_ready(settings: Settings | None = None) -> bool:
    return all(p.exists() for p in model_paths(settings))


def _download(url: str, dest: Path, on_progress: ProgressCallback | None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (URL fija, https)
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if on_progress:
                on_progress(done, total)
    tmp.replace(dest)


def ensure_models(settings: Settings | None = None, on_progress: ProgressCallback | None = None) -> None:
    """Descarga una sola vez los modelos de segmentación y de huella de voz (~45 MB)."""
    seg, emb = model_paths(settings)
    d = diarization_dir(settings)
    if not seg.exists():
        archive = d / "segmentation.tar.bz2"
        _download(SEGMENTATION_ARCHIVE, archive, on_progress)
        with tarfile.open(archive) as tar:
            member = tar.getmember(SEGMENTATION_FILE)
            member.name = SEGMENTATION_FILE  # sin rutas absolutas ni ".." del archivo
            tar.extract(member, d, filter="data")
        archive.unlink()
    if not emb.exists():
        _download(EMBEDDING_URL, emb, on_progress)


def _read_audio(wav_path: str | Path) -> np.ndarray:
    """Lee el WAV mono 16 kHz a float32 en un único array, por bloques."""
    with wave.open(str(wav_path)) as w:
        total = w.getnframes()
        audio = np.empty(total, dtype=np.float32)
        pos = 0
        while pos < total:
            block = np.frombuffer(w.readframes(min(READ_BLOCK, total - pos)), dtype=np.int16)
            audio[pos : pos + len(block)] = block.astype(np.float32) / 32768.0
            pos += len(block)
    return audio


def clean_turns(turns: list[Turn], min_seconds: float = MIN_TURN_SECONDS) -> list[Turn]:
    """Funde los turnos muy cortos con el vecino y une turnos contiguos del mismo hablante."""
    turns = sorted(turns, key=lambda t: t.start)
    out: list[Turn] = []
    for i, t in enumerate(turns):
        if t.end - t.start < min_seconds and (out or i + 1 < len(turns)):
            if out:  # pasa a manos del turno anterior
                out[-1] = Turn(out[-1].start, max(out[-1].end, t.end), out[-1].speaker_id)
            else:  # es el primero: se lo queda el siguiente
                turns[i + 1] = Turn(t.start, turns[i + 1].end, turns[i + 1].speaker_id)
            continue
        if out and out[-1].speaker_id == t.speaker_id:
            out[-1] = Turn(out[-1].start, max(out[-1].end, t.end), t.speaker_id)
        else:
            out.append(t)
    return out


def absorb_minor_speakers(
    turns: list[Turn], min_share: float = MINOR_SPEAKER_SHARE, min_seconds: float = MINOR_SPEAKER_SECONDS
) -> list[Turn]:
    """Reasigna los hablantes con muy poco habla al hablante principal más cercano en el tiempo y
    renumera "Persona N" por orden de aparición. Si nadie supera el mínimo, no toca nada."""
    turns = sorted(turns, key=lambda t: t.start)
    talk: dict[str, float] = {}
    for t in turns:
        talk[t.speaker_id] = talk.get(t.speaker_id, 0.0) + t.end - t.start
    floor = max(min_seconds, min_share * sum(talk.values()))
    major = {s for s, d in talk.items() if d >= floor}
    if not major or len(major) == len(talk):
        return turns
    merged = []
    for i, t in enumerate(turns):
        if t.speaker_id in major:
            merged.append(t)
            continue
        before = next((x for x in reversed(turns[:i]) if x.speaker_id in major), None)
        after = next((x for x in turns[i + 1 :] if x.speaker_id in major), None)
        near = min(
            (x for x in (before, after) if x),
            key=lambda x: abs(t.start - x.end) if x is before else abs(x.start - t.end),
        )
        merged.append(Turn(t.start, t.end, near.speaker_id))
    order: dict[str, str] = {}
    renamed = [Turn(t.start, t.end, order.setdefault(t.speaker_id, f"Persona {len(order) + 1}")) for t in merged]
    return clean_turns(renamed)


def diarize(
    wav_path: str | Path,
    num_speakers: int | None = None,
    on_progress: ProgressCallback | None = None,
    settings: Settings | None = None,
) -> list[Turn]:
    """Devuelve los turnos de habla de un WAV mono 16 kHz con ids "Persona N"."""
    import sherpa_onnx  # import perezoso

    settings = settings or get_settings()
    if not models_ready(settings):
        raise FileNotFoundError("Faltan los modelos de voz. Ejecuta: transcriptor models --descargar-voz")
    seg_model, emb_model = model_paths(settings)
    threads = max(1, min(8, (os.cpu_count() or 4)))
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_model)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_model), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers or -1, threshold=CLUSTER_THRESHOLD
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    sd = sherpa_onnx.OfflineSpeakerDiarization(config)
    audio = _read_audio(wav_path)

    def callback(done: int, total: int) -> int:
        if on_progress:
            on_progress(done, total)
        return 0

    result = sd.process(audio, callback=callback).sort_by_start_time()
    order: dict[int, str] = {}  # numera "Persona N" por orden de aparición
    turns = [
        Turn(r.start, r.end, order.setdefault(r.speaker, f"Persona {len(order) + 1}")) for r in result
    ]
    turns = clean_turns(turns)
    # Con --num-speakers el usuario ya fijó la cantidad: no se descarta ningún hablante
    return turns if num_speakers else absorb_minor_speakers(turns)


def _overlaps(seg_start: float, seg_end: float, turns: list[Turn], starts: list[float]) -> dict[str, float]:
    """Segundos de solape del intervalo con cada hablante."""
    acc: dict[str, float] = {}
    i = max(0, bisect.bisect_right(starts, seg_start) - 1)
    while i < len(turns) and turns[i].start < seg_end:
        o = min(seg_end, turns[i].end) - max(seg_start, turns[i].start)
        if o > 0:
            acc[turns[i].speaker_id] = acc.get(turns[i].speaker_id, 0.0) + o
        i += 1
    return acc


def _nearest(t: float, turns: list[Turn]) -> str | None:
    if not turns:
        return None
    return min(turns, key=lambda x: 0.0 if x.start <= t <= x.end else min(abs(t - x.start), abs(t - x.end))).speaker_id


def assign_speakers(transcript: Transcript, turns: list[Turn], split_ratio: float = 0.6) -> Transcript:
    """Asigna a cada segmento el hablante con mayor solape. Si el segmento cruza dos hablantes casi
    por igual y hay palabras con tiempos, lo divide en dos. Devuelve el mismo Transcript."""
    turns = sorted(turns, key=lambda t: t.start)
    starts = [t.start for t in turns]
    result: list[Segment] = []
    for seg in transcript.segments:
        ov = _overlaps(seg.start, seg.end, turns, starts)
        ranked = sorted(ov.items(), key=lambda kv: kv[1], reverse=True)
        if len(ranked) >= 2 and seg.words and ranked[1][1] >= split_ratio * ranked[0][1]:
            result.extend(_split_by_words(seg, turns, starts))
            continue
        seg.speaker = ranked[0][0] if ranked else _nearest((seg.start + seg.end) / 2, turns)
        result.append(seg)
    transcript.segments = result
    return transcript


def _split_by_words(seg: Segment, turns: list[Turn], starts: list[float]) -> list[Segment]:
    groups: list[tuple[str | None, list[dict]]] = []
    for w in seg.words or []:
        ov = _overlaps(w["start"], w["end"], turns, starts)
        who = max(ov, key=ov.get) if ov else _nearest((w["start"] + w["end"]) / 2, turns)
        if groups and groups[-1][0] == who:
            groups[-1][1].append(w)
        else:
            groups.append((who, [w]))
    if len(groups) < 2:
        seg.speaker = groups[0][0] if groups else None
        return [seg]
    return [
        Segment(
            start=ws[0]["start"],
            end=ws[-1]["end"],
            text="".join(w["word"] for w in ws).strip(),
            speaker=who,
            words=ws,
        )
        for who, ws in groups
    ]
