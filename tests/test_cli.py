import pytest
from typer.testing import CliRunner

from transcriptor import cli
from transcriptor.models import Segment, Transcript

runner = CliRunner()


class FakeEngine:
    def transcribe(self, wav, language=None, on_progress=None, partial_path=None, **_):
        if on_progress:
            on_progress(5.0, 5.0)
        return Transcript(language or "es", 5.0, [Segment(0, 2, "Hola mundo")])


@pytest.fixture(autouse=True)
def fake_engine(monkeypatch):
    monkeypatch.setattr(cli.TranscriptionEngine, "get", classmethod(lambda cls, s=None: FakeEngine()))


def test_help_documenta_opciones():
    r = runner.invoke(cli.app, ["transcribe", "--help"])
    assert r.exit_code == 0
    for opt in ("--format", "--language", "--model", "--output", "--device", "--stdout"):
        assert opt in r.output


def test_srt_junto_al_video(media_dir, tmp_path):
    video = tmp_path / "reunion.mp4"
    video.write_bytes((media_dir / "video.mp4").read_bytes())
    r = runner.invoke(cli.app, ["transcribe", str(video), "-f", "srt"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "reunion.srt").read_text(encoding="utf-8").startswith("1\n00:00:00,000 --> 00:00:02,000")
    assert not (tmp_path / "reunion.partial.jsonl").exists()


def test_varios_formatos_y_carpeta(media_dir, tmp_path):
    (tmp_path / "a.mp3").write_bytes((media_dir / "tono.mp3").read_bytes())
    (tmp_path / "nota.txt").write_text("no es medio")
    out = tmp_path / "salida"
    r = runner.invoke(cli.app, ["transcribe", str(tmp_path), "-f", "txt", "-f", "vtt", "-o", str(out)])
    assert r.exit_code == 0, r.output
    assert (out / "a.txt").exists() and (out / "a.vtt").exists()


def test_stdout(media_dir):
    r = runner.invoke(cli.app, ["transcribe", str(media_dir / "tono.mp3"), "--stdout"])
    assert r.exit_code == 0
    assert "Hola mundo" in r.stdout


def test_archivo_invalido_sale_con_1(tmp_path):
    falso = tmp_path / "falso.mp4"
    falso.write_text("no soy un video")
    r = runner.invoke(cli.app, ["transcribe", str(falso)])
    assert r.exit_code == 1


def test_formato_invalido_sale_con_1(media_dir):
    r = runner.invoke(cli.app, ["transcribe", str(media_dir / "tono.mp3"), "-f", "pdf"])
    assert r.exit_code == 1


def test_error_interno_sale_con_2(media_dir, monkeypatch):
    class Roto:
        def transcribe(self, *a, **k):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli.TranscriptionEngine, "get", classmethod(lambda cls, s=None: Roto()))
    r = runner.invoke(cli.app, ["transcribe", str(media_dir / "tono.mp3")])
    assert r.exit_code == 2


def test_models_lista():
    r = runner.invoke(cli.app, ["models"])
    assert r.exit_code == 0
    assert "large-v3" in r.output and "small" in r.output


@pytest.fixture
def isolated_models_dir(tmp_path, monkeypatch):
    s = cli.get_settings().model_copy(update={"models_dir": tmp_path / "cfg"})
    monkeypatch.setattr(cli, "get_settings", lambda: s)
    return tmp_path / "cfg"


def test_pregunta_la_ruta_y_genera_txt_y_srt(media_dir, tmp_path, isolated_models_dir):
    video = tmp_path / "reunion.mp4"
    video.write_bytes((media_dir / "video.mp4").read_bytes())
    r = runner.invoke(cli.app, [], input=f"{video}\nn\n\n")  # ruta, sin hablantes, Enter = sin resumen
    assert r.exit_code == 0, r.output
    assert "¿Dónde está el video" in r.output
    assert (tmp_path / "reunion.txt").exists() and (tmp_path / "reunion.srt").exists()
    assert (isolated_models_dir / "ultima_ruta.txt").read_text(encoding="utf-8") == str(video)


def test_ruta_con_comillas_y_reintento_si_no_existe(media_dir, tmp_path, isolated_models_dir):
    video = tmp_path / "con espacios.mp4"
    video.write_bytes((media_dir / "video.mp4").read_bytes())
    r = runner.invoke(cli.app, [], input=f'{tmp_path / "no_existe.mp4"}\n"{video}"\nn\n\n')
    assert r.exit_code == 0, r.output
    assert "No existe" in r.output
    assert (tmp_path / "con espacios.srt").exists()


def test_pregunta_con_carpeta(media_dir, tmp_path, isolated_models_dir):
    carpeta = tmp_path / "reuniones"
    carpeta.mkdir()
    (carpeta / "a.mp3").write_bytes((media_dir / "tono.mp3").read_bytes())
    r = runner.invoke(cli.app, [], input=f"{carpeta}\nn\n\n")
    assert r.exit_code == 0, r.output
    assert (carpeta / "a.srt").exists()


def test_pregunta_el_resumen_y_genera_el_extractivo(media_dir, tmp_path, isolated_models_dir):
    video = tmp_path / "reunion.mp4"
    video.write_bytes((media_dir / "video.mp4").read_bytes())
    r = runner.invoke(cli.app, [], input=f"{video}\nn\nextractivo\n")
    assert r.exit_code == 0, r.output
    assert "¿Generar resumen?" in r.output
    md = (tmp_path / "reunion.resumen.md").read_text(encoding="utf-8")
    assert md.startswith("# Resumen: reunion") and "Resumen extractivo (sin IA)" in md


def test_resumir_extractivo_sobre_transcripcion_existente(tmp_path):
    srt = tmp_path / "r.srt"
    srt.write_text("1\n00:00:00,000 --> 00:00:04,000\nHay que enviar el informe del proyecto para el viernes.\n", encoding="utf-8")
    r = runner.invoke(cli.app, ["resumir", str(srt), "--modo", "extractivo"])
    assert r.exit_code == 0, r.output
    assert "enviar el informe" in (tmp_path / "r.resumen.md").read_text(encoding="utf-8")


def test_resumir_modo_invalido_sale_con_1(tmp_path):
    f = tmp_path / "r.srt"
    f.write_text("", encoding="utf-8")
    assert runner.invoke(cli.app, ["resumir", str(f), "--modo", "magia"]).exit_code == 1
