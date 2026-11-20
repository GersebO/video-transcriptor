"""Extracción de audio con ffmpeg (video/audio -> WAV mono 16 kHz)."""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .config import get_settings


class FFmpegNotFoundError(RuntimeError):
    """ffmpeg no está instalado o no está en el PATH."""


class InvalidMediaError(ValueError):
    """El archivo no tiene pista de audio o está corrupto."""


class InsufficientDiskError(OSError):
    """No hay espacio libre suficiente para extraer el audio."""


# WAV mono 16 kHz de 16 bits = 32 000 bytes por segundo
_WAV_BYTES_PER_SECOND = 16000 * 2


def _find_binary(name: str) -> str | None:
    """Busca el ejecutable en el PATH. En Windows, si no está, relee el PATH del sistema y del
    usuario: una terminal abierta antes de instalar ffmpeg no ve el PATH nuevo."""
    found = shutil.which(name)
    if found or sys.platform != "win32":
        return found
    try:
        import winreg

        parts = []
        for root, sub in (
            (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            (winreg.HKEY_CURRENT_USER, "Environment"),
        ):
            with winreg.OpenKey(root, sub) as key:
                parts.append(os.path.expandvars(winreg.QueryValueEx(key, "Path")[0]))
    except OSError:
        return None
    return shutil.which(name, path=os.pathsep.join(parts))


def check_ffmpeg() -> str:
    path = _find_binary("ffmpeg")
    if path is None:
        raise FFmpegNotFoundError(
            "No se encontró ffmpeg. Instálalo (https://ffmpeg.org) y asegúrate de que esté en el PATH."
        )
    return path


def probe_duration(input_path: Path) -> float:
    """Duración en segundos usando ffprobe; lanza InvalidMediaError si no hay audio."""
    ffprobe = _find_binary("ffprobe")
    if ffprobe is None:
        raise FFmpegNotFoundError("No se encontró ffprobe (viene con ffmpeg).")
    result = subprocess.run(
        [
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "format=duration", "-of", "default=nw=1:nk=1",
            str(input_path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise InvalidMediaError(
            f"El archivo no es un medio válido o no contiene audio: {input_path.name}"
        ) from None


def extract_audio(input_path: str | Path, out_dir: str | Path | None = None) -> Path:
    """Convierte el medio a WAV mono 16 kHz y devuelve la ruta del WAV.

    Sin límite de tamaño: ffmpeg lee y escribe en streaming. Para audios muy largos
    se usa `-rf64 auto` (el WAV normal se limita a 4 GB, ~35 h).
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise InvalidMediaError(f"No existe el archivo: {input_path}")
    ffmpeg = check_ffmpeg()

    duration = probe_duration(input_path)

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="transcriptor_", dir=get_settings().temp_dir or None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    needed = int(duration * _WAV_BYTES_PER_SECOND * 1.05)
    free = shutil.disk_usage(out_dir).free
    if free < needed:
        raise InsufficientDiskError(
            f"Espacio insuficiente en {out_dir}: se necesitan ~{needed // 2**20} MB "
            f"y hay {free // 2**20} MB libres. Usa TRANSCRIPTOR_TEMP_DIR para elegir otro disco."
        )

    wav_path = out_dir / f"{input_path.stem}.wav"
    result = subprocess.run(
        [
            ffmpeg, "-y", "-nostdin", "-v", "error", "-i", str(input_path),
            "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", "-rf64", "auto", str(wav_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not wav_path.exists():
        raise InvalidMediaError(
            f"ffmpeg no pudo procesar {input_path.name}: {result.stderr.strip()[:300]}"
        )
    return wav_path
