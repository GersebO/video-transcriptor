import wave

import pytest

from transcriptor import audio
from transcriptor.audio import InvalidMediaError, extract_audio


@pytest.mark.parametrize("name", ["video.mp4", "tono.mp3"])
def test_extract_audio_produce_wav_mono_16k(media_dir, tmp_path, name):
    wav = extract_audio(media_dir / name, tmp_path)
    with wave.open(str(wav)) as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == 16000
        assert 4.5 < w.getnframes() / 16000 < 5.5


def test_archivo_de_texto_renombrado_falla(tmp_path):
    falso = tmp_path / "falso.mp4"
    falso.write_text("esto no es un video")
    with pytest.raises(InvalidMediaError):
        extract_audio(falso, tmp_path / "out")


def test_archivo_inexistente(tmp_path):
    with pytest.raises(InvalidMediaError):
        extract_audio(tmp_path / "no_existe.mp4")


def test_sin_ffmpeg(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "_find_binary", lambda _: None)
    f = tmp_path / "x.mp4"
    f.write_bytes(b"x")
    with pytest.raises(audio.FFmpegNotFoundError):
        extract_audio(f)
