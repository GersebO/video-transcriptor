import shutil
import subprocess

import pytest

from transcriptor.config import Settings
from transcriptor.engine import TranscriptionEngine
from transcriptor.models import Segment, Transcript


def test_transcript_text():
    t = Transcript("es", 3.0, [Segment(0, 1, " Hola"), Segment(1, 2, "mundo ")])
    assert t.text == "Hola mundo"


@pytest.mark.slow
def test_transcribe_real_y_modelo_se_carga_una_vez(tmp_path):
    espeak = shutil.which("espeak") or shutil.which("espeak-ng")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg no instalado")
    wav = tmp_path / "voz.wav"
    if espeak:
        subprocess.run([espeak, "-v", "es", "-w", str(wav), "hola esto es una prueba"], check=True)
    else:
        # Sin sintetizador: tono de 3 s; solo se comprueba que corre sin error
        subprocess.run([ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i",
                        "sine=frequency=440:duration=3", "-ar", "16000", "-ac", "1", str(wav)], check=True)

    settings = Settings(model_size="tiny", models_dir=tmp_path / "models")
    engine = TranscriptionEngine(settings)
    progress = []
    partial = tmp_path / "parcial.jsonl"
    t1 = engine.transcribe(wav, language="es", on_progress=lambda a, b: progress.append(a / b),
                           partial_path=partial)
    model_first = engine._model
    engine.transcribe(wav, language="es")
    assert engine._model is model_first  # no se recarga
    assert progress and progress[-1] == 1.0
    assert t1.language == "es"
    if espeak:
        assert t1.text


def test_troceado_corta_en_silencio_y_cubre_todo_el_audio(tmp_path):
    import wave

    import numpy as np

    from transcriptor.engine import iter_audio_chunks

    sr = 16000
    t = np.arange(35 * sr) / sr
    audio = (np.sin(2 * np.pi * 440 * t) * 8000).astype(np.int16)
    audio[8 * sr : 9 * sr] = 0  # silencio entre 8 s y 9 s
    wav = tmp_path / "x.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())

    chunks = list(iter_audio_chunks(wav, chunk_seconds=10))
    assert sum(len(c) for _, c in chunks) == len(audio)  # no se pierde ni se repite nada
    assert 8.0 <= chunks[0][1].shape[0] / sr <= 9.0  # primer corte dentro del silencio
    assert all(abs(start - sum(len(c) for _, c in chunks[:i]) / sr) < 1e-6 for i, (start, _) in enumerate(chunks))
