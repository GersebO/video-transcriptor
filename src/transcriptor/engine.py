"""Motor de transcripción basado en faster-whisper (carga perezosa, un solo modelo)."""

import json
import threading
import wave
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .config import Settings, get_settings
from .models import Segment, Transcript

ProgressCallback = Callable[[float, float], None]  # (segundos procesados, duración total)

# faster-whisper decodifica todo el audio a float32 en memoria (~800 MB por hora con VAD),
# así que las reuniones largas se procesan por bloques para mantener la memoria acotada.
CHUNK_SECONDS = 20 * 60
SEARCH_SECONDS = 60  # el corte se busca en el punto más silencioso de este tramo final


def iter_audio_chunks(wav_path, chunk_seconds: int = CHUNK_SECONDS) -> Iterator[tuple[float, np.ndarray]]:
    """Genera (inicio_en_segundos, audio float32) cortando cada bloque en un silencio."""
    with wave.open(str(wav_path)) as w:
        sr, total = w.getframerate(), w.getnframes()
        chunk, search = chunk_seconds * sr, SEARCH_SECONDS * sr
        pos = 0
        while pos < total:
            audio = np.frombuffer(w.readframes(min(chunk, total - pos)), dtype=np.int16)
            if pos + len(audio) < total:  # no es el último: cortar donde haya menos energía
                win = sr // 2
                tail = audio[-min(search, len(audio)):]
                k = len(tail) // win
                if k:
                    energy = np.abs(tail[: k * win].astype(np.float32)).reshape(k, win).mean(axis=1)
                    audio = audio[: len(audio) - len(tail) + int(energy.argmin()) * win + win // 2]
            yield pos / sr, audio.astype(np.float32) / 32768.0
            pos += len(audio)
            w.setpos(pos)


class TranscriptionEngine:
    _instances: dict[tuple, "TranscriptionEngine"] = {}
    _lock = threading.Lock()

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._model = None
        self._model_lock = threading.Lock()

    @classmethod
    def get(cls, settings: Settings | None = None) -> "TranscriptionEngine":
        """Devuelve la instancia compartida para esta configuración de modelo."""
        s = settings or get_settings()
        key = (s.model_size, s.device, s.compute_type, str(s.models_dir))
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(s)
            return cls._instances[key]

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        with self._model_lock:
            if self._model is None:
                from faster_whisper import WhisperModel  # import perezoso: es pesado

                self.settings.models_dir.mkdir(parents=True, exist_ok=True)
                self._model = WhisperModel(
                    self.settings.model_size,
                    device=self.settings.device,
                    compute_type=self.settings.compute_type,
                    download_root=str(self.settings.models_dir),
                )
        return self._model

    def load(self) -> None:
        """Fuerza la carga del modelo (útil al arrancar la API)."""
        self._load()

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        word_timestamps: bool = False,
        on_progress: ProgressCallback | None = None,
        partial_path: str | Path | None = None,
    ) -> Transcript:
        """Transcribe un WAV mono 16 kHz por bloques (memoria acotada, sin límite de duración).
        Los segmentos se consumen uno a uno y, si se indica `partial_path`, se van volcando a un
        .jsonl para no perder el avance en reuniones largas si el proceso se interrumpe."""
        model = self._load()
        language = language or self.settings.default_language or None
        with wave.open(str(audio_path)) as w:
            duration = w.getnframes() / w.getframerate()

        transcript = Transcript(language=language or "", duration=duration)
        partial = open(partial_path, "w", encoding="utf-8") if partial_path else None
        try:
            for offset, audio in iter_audio_chunks(audio_path):
                segments_iter, info = model.transcribe(
                    audio,
                    language=language,
                    word_timestamps=word_timestamps,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                    condition_on_previous_text=False,  # evita bucles/alucinaciones en audios largos
                )
                found = False
                for seg in segments_iter:
                    found = True
                    words = [
                        {"start": offset + w.start, "end": offset + w.end, "word": w.word}
                        for w in (seg.words or [])
                    ] or None
                    segment = Segment(
                        start=offset + seg.start, end=offset + seg.end, text=seg.text.strip(), words=words
                    )
                    transcript.segments.append(segment)
                    if partial:
                        partial.write(json.dumps(segment.to_dict(), ensure_ascii=False) + "\n")
                        partial.flush()
                    if on_progress:
                        on_progress(min(segment.end, duration), duration)
                if not transcript.language or (language is None and found):
                    transcript.language = info.language
                if language is None and found:
                    language = info.language  # fija el idioma detectado para el resto de bloques
                if on_progress:
                    on_progress(min(offset + len(audio) / 16000, duration), duration)
        finally:
            if partial:
                partial.close()
        if on_progress:
            on_progress(duration, duration)
        return transcript
