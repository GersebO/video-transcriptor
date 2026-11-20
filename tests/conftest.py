import shutil
import subprocess

import pytest


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory):
    """Genera con ffmpeg un MP3 y un MP4 de 5 s (tono) en lugar de versionar binarios."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg no instalado")
    d = tmp_path_factory.mktemp("media")
    base = [ffmpeg, "-y", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=5"]
    subprocess.run(base + [str(d / "tono.mp3")], check=True)
    subprocess.run(
        base + ["-f", "lavfi", "-i", "color=c=blue:s=160x120:d=5", "-shortest", str(d / "video.mp4")],
        check=True,
    )
    return d
