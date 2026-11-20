"""Exportación de un Transcript a txt, srt, vtt y json."""

import json
import re
from pathlib import Path

from .models import Segment, Transcript

FORMATS = ("txt", "srt", "vtt", "json")


def _timestamp(seconds: float, sep: str) -> str:
    # Las horas no se limitan a 2 dígitos: soporta reuniones de más de 99 h
    total_ms = max(0, round(seconds * 1000))
    h, rest = divmod(total_ms, 3_600_000)
    m, rest = divmod(rest, 60_000)
    s, ms = divmod(rest, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_txt(t: Transcript) -> str:
    if not t.has_speakers:
        return "\n".join(s.text.strip() for s in t.segments if s.text.strip()) + "\n"
    # Las líneas seguidas del mismo hablante se agrupan bajo un solo "Nombre: ..."
    runs: list[tuple[str | None, list[str]]] = []
    for s in t.segments:
        text = s.text.strip()
        if not text:
            continue
        who = t.label(s.speaker)
        if runs and runs[-1][0] == who:
            runs[-1][1].append(text)
        else:
            runs.append((who, [text]))
    return "\n".join(f"{who}: {' '.join(p)}" if who else " ".join(p) for who, p in runs) + "\n"


def to_srt(t: Transcript) -> str:
    blocks = []
    for i, s in enumerate(t.segments, 1):
        who = t.label(s.speaker)
        text = f"[{who}] {s.text.strip()}" if who else s.text.strip()
        blocks.append(f"{i}\n{_timestamp(s.start, ',')} --> {_timestamp(s.end, ',')}\n{text}\n")
    return "\n".join(blocks)


def to_vtt(t: Transcript) -> str:
    blocks = ["WEBVTT\n"]
    for s in t.segments:
        who = t.label(s.speaker)
        text = f"<v {who}>{s.text.strip()}" if who else s.text.strip()
        blocks.append(f"{_timestamp(s.start, '.')} --> {_timestamp(s.end, '.')}\n{text}\n")
    return "\n".join(blocks)


def to_json(t: Transcript) -> str:
    data = {
        "language": t.language,
        "duration": t.duration,
        "text": t.text,
        "segments": [s.to_dict() for s in t.segments],
    }
    if t.speakers:
        data["speakers"] = t.speakers
    return json.dumps(data, ensure_ascii=False, indent=2)


_EXPORTERS = {"txt": to_txt, "srt": to_srt, "vtt": to_vtt, "json": to_json}


def export(t: Transcript, fmt: str) -> str:
    try:
        return _EXPORTERS[fmt](t)
    except KeyError:
        raise ValueError(f"Formato no soportado: {fmt}. Usa uno de: {', '.join(FORMATS)}") from None


_SRT_TIME = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d+):(\d{2}):(\d{2})[,.](\d{3})")


def parse_srt(text: str) -> Transcript:
    """Reconstruye un Transcript (sin idioma ni palabras) a partir de un .srt."""
    segments = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip().lstrip("﻿")):
        lines = block.strip().splitlines()
        for i, line in enumerate(lines):
            m = _SRT_TIME.search(line)
            if m:
                h1, m1, s1, ms1, h2, m2, s2, ms2 = (int(x) for x in m.groups())
                body = " ".join(x.strip() for x in lines[i + 1 :] if x.strip())
                segments.append(
                    Segment(
                        h1 * 3600 + m1 * 60 + s1 + ms1 / 1000,
                        h2 * 3600 + m2 * 60 + s2 + ms2 / 1000,
                        body,
                    )
                )
                break
    return Transcript(language="", duration=segments[-1].end if segments else 0.0, segments=segments)


def load_transcript(path: str | Path) -> Transcript:
    """Carga una transcripción desde .json (completa) o .srt (solo texto y tiempos)."""
    path = Path(path)
    raw = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".srt":
        return parse_srt(raw)
    return Transcript.from_dict(json.loads(raw))
