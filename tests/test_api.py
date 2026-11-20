import time

import pytest
from fastapi.testclient import TestClient

from transcriptor.api import create_app
from transcriptor.config import Settings
from transcriptor.models import Segment, Transcript


class FakeEngine:
    loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, wav, language=None, on_progress=None, **_):
        if on_progress:
            on_progress(2.5, 5.0)
            on_progress(5.0, 5.0)
        return Transcript(language or "es", 5.0, [Segment(0, 2, "Hola mundo")])


def make_settings(tmp_path, **kw):
    return Settings(jobs_db=tmp_path / "data" / "jobs.db", **kw)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path), FakeEngine())) as c:
        yield c


def wait_done(client, job_id, timeout=20):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/transcriptions/{job_id}").json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError("el trabajo no terminó")


def upload(client, path, **data):
    with open(path, "rb") as f:
        return client.post("/transcriptions", files={"file": (path.name, f)}, data=data)


def test_health_con_modelo_cargado_al_arrancar(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_loaded"] is True


def test_flujo_completo(client, media_dir):
    r = upload(client, media_dir / "video.mp4")
    assert r.status_code == 202
    job_id = r.json()["job_id"]
    body = wait_done(client, job_id)
    assert body["status"] == "done" and body["progress"] == 1.0
    assert body["result"]["segments"][0]["text"] == "Hola mundo"

    srt = client.get(f"/transcriptions/{job_id}/download", params={"format": "srt"})
    assert srt.status_code == 200
    assert srt.text.startswith("1\n00:00:00,000 --> 00:00:02,000")
    assert 'filename="video.srt"' in srt.headers["content-disposition"]

    assert client.delete(f"/transcriptions/{job_id}").status_code == 204
    assert client.get(f"/transcriptions/{job_id}").status_code == 404


def test_descarga_antes_de_terminar_da_409(tmp_path, media_dir):
    import threading

    gate = threading.Event()

    class Lento(FakeEngine):
        def transcribe(self, *a, **k):
            gate.wait(10)
            return super().transcribe(*a, **k)

    with TestClient(create_app(make_settings(tmp_path), Lento())) as c:
        job_id = upload(c, media_dir / "tono.mp3").json()["job_id"]
        r = c.get(f"/transcriptions/{job_id}/download")
        assert r.status_code == 409 and r.json()["error"]["code"] == "not_finished"
        gate.set()
        wait_done(c, job_id)


def test_archivo_invalido_deja_el_trabajo_en_error(client, tmp_path):
    falso = tmp_path / "falso.mp4"
    falso.write_text("no soy un video")
    body = wait_done(client, upload(client, falso).json()["job_id"])
    assert body["status"] == "error"
    assert body["error"]["code"] == "invalid_media"


def test_sin_limite_por_defecto_y_413_si_se_configura(tmp_path, media_dir):
    grande = tmp_path / "grande.bin"
    grande.write_bytes(b"\0" * (3 * 1024 * 1024))

    with TestClient(create_app(make_settings(tmp_path / "a"), FakeEngine())) as c:
        assert upload(c, grande).status_code == 202  # max_upload_mb=0 -> ilimitado

    with TestClient(create_app(make_settings(tmp_path / "b", max_upload_mb=1), FakeEngine())) as c:
        r = upload(c, grande)
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "file_too_large"


def test_formato_invalido_y_404(client):
    assert client.get("/transcriptions/no-existe").json()["error"]["code"] == "job_not_found"
    assert client.get("/transcriptions/x/download", params={"format": "pdf"}).status_code == 400


def test_trabajo_pendiente_se_reencola_tras_reinicio(tmp_path, media_dir):
    settings = make_settings(tmp_path)
    # Simula un servidor que murió con un trabajo "processing" y su subida en disco
    from transcriptor.jobs import JobManager

    m = JobManager(settings, FakeEngine())
    upload_path = m.new_upload_path("abc", "tono.mp3")
    upload_path.write_bytes((media_dir / "tono.mp3").read_bytes())
    m.store.create("abc", "tono.mp3", None, upload_path)
    m.store.update("abc", status="processing")
    m.shutdown()

    with TestClient(create_app(settings, FakeEngine())) as c:
        assert wait_done(c, "abc")["status"] == "done"
