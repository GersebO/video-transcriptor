"""Nombres de los hablantes: archivo .speakers.json, frases de ejemplo y muestras de voz."""

import json
import subprocess
from pathlib import Path
from typing import Callable

from .audio import _find_binary
from .models import Transcript

EXAMPLE_MIN_CHARS = 25  # las frases demasiado cortas ("sí", "vale") no ayudan a reconocer una voz


def names_path(transcript_path: str | Path) -> Path:
    """reunion.json -> reunion.speakers.json"""
    p = Path(transcript_path)
    return p.with_name(f"{p.stem}.speakers.json")


def load_names(transcript_path: str | Path) -> dict[str, str]:
    try:
        names = json.loads(names_path(transcript_path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in names.items() if isinstance(v, str) and v.replace("﻿", "").strip()}


def save_names(transcript_path: str | Path, names: dict[str, str]) -> Path:
    path = names_path(transcript_path)
    path.write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def speaker_ids(t: Transcript) -> list[str]:
    """Ids de hablante en orden de aparición."""
    seen: dict[str, None] = {}
    for s in t.segments:
        if s.speaker:
            seen.setdefault(s.speaker)
    return list(seen)


def examples(t: Transcript, speaker: str, n: int = 3) -> list[tuple[float, str]]:
    """n frases (tiempo, texto) del hablante, repartidas a lo largo de la reunión."""
    own = [s for s in t.segments if s.speaker == speaker]
    good = [s for s in own if len(s.text.strip()) >= EXAMPLE_MIN_CHARS] or own
    if len(good) <= n:
        picked = good
    else:
        picked = [good[round(i * (len(good) - 1) / (n - 1))] for i in range(n)] if n > 1 else good[:1]
    return [(s.start, s.text.strip()) for s in picked]


def play_sample(media_path: str | Path, start: float, seconds: int = 5) -> bool:
    """Reproduce unos segundos con ffplay. Devuelve False si ffplay no está disponible."""
    ffplay = _find_binary("ffplay")
    if not ffplay:
        return False
    subprocess.run(
        [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet", "-ss", f"{start:.1f}", "-t", str(seconds), str(media_path)],
        check=False,
    )
    return True


def _clock(seconds: float) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def ask_names(
    t: Transcript,
    media_path: str | Path | None,
    ask: Callable[[str, str], str],
    say: Callable[[str], None],
) -> dict[str, str]:
    """Pregunta el nombre de cada hablante. `ask(pregunta, por_defecto)` devuelve lo que escribió el
    usuario; con 'r' vuelve a reproducir la voz. Enter deja el nombre que ya tenía."""
    for sid in speaker_ids(t):
        shown = examples(t, sid)
        say(f"\n[bold]{sid}[/bold]")
        for start, text in shown:
            say(f"  [{_clock(start)}] {text}")
        current = t.speakers.get(sid, sid)
        sample = shown[0][0] if shown else None
        while True:
            hint = " (r = escuchar 5 s)" if media_path and sample is not None else ""
            # El BOM aparece al redirigir entrada desde PowerShell y no cuenta como espacio para strip()
            answer = ask(f"Nombre de {sid}{hint}", current).replace("﻿", "").strip()
            if answer.lower() == "r" and media_path and sample is not None:
                if not play_sample(media_path, sample):
                    say("[yellow]ffplay no está disponible; no puedo reproducir la voz.[/yellow]")
                continue
            break
        if answer and answer != sid:
            t.speakers[sid] = answer
        else:
            t.speakers.pop(sid, None)
    return t.speakers
