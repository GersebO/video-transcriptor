"""Cola de trabajos persistente (SQLite) con un worker en segundo plano.

Pensada para reuniones largas: el estado y los resultados viven en disco, así que
sobreviven a un reinicio del servidor y no ocupan memoria.
"""

import json
import shutil
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audio import extract_audio
from .config import Settings
from .engine import TranscriptionEngine
from .formats import to_json
from .models import Transcript

PENDING, PROCESSING, DONE, ERROR = "pending", "processing", "done", "error"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    filename TEXT NOT NULL,
    language TEXT,
    upload_path TEXT,
    result_path TEXT,
    error_code TEXT,
    error_message TEXT,
    created_at REAL NOT NULL
)
"""


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(_SCHEMA)
            self._conn.commit()

    def _exec(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def create(self, job_id: str, filename: str, language: str | None, upload_path: Path) -> None:
        self._exec(
            "INSERT INTO jobs (id, status, filename, language, upload_path, created_at) VALUES (?,?,?,?,?,?)",
            (job_id, PENDING, filename, language, str(upload_path), time.time()),
        )

    def get(self, job_id: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    def update(self, job_id: str, **fields) -> None:
        cols = ", ".join(f"{k} = ?" for k in fields)
        self._exec(f"UPDATE jobs SET {cols} WHERE id = ?", (*fields.values(), job_id))

    def delete(self, job_id: str) -> None:
        self._exec("DELETE FROM jobs WHERE id = ?", (job_id,))

    def unfinished(self) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM jobs WHERE status IN (?,?) ORDER BY created_at", (PENDING, PROCESSING)
            ).fetchall()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class JobManager:
    def __init__(self, settings: Settings, engine: TranscriptionEngine):
        self.settings = settings
        self.engine = engine
        self.store = JobStore(settings.jobs_db)
        base = settings.jobs_db.parent
        self.results_dir = base / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Las subidas van al temp_dir configurado (disco grande) o junto a la base de datos
        self.uploads_dir = (Path(settings.temp_dir) if settings.temp_dir else base) / "uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="transcriptor-worker")
        self._recover()

    def _recover(self) -> None:
        """Tras un reinicio: reencola lo pendiente/en proceso si la subida sigue en disco."""
        for row in self.store.unfinished():
            upload = Path(row["upload_path"] or "")
            if upload.is_file():
                self.store.update(row["id"], status=PENDING, progress=0)
                self._pool.submit(self._run, row["id"])
            else:
                self.store.update(
                    row["id"], status=ERROR, error_code="interrupted",
                    error_message="El servidor se reinició y el archivo subido ya no está disponible.",
                )

    def new_upload_path(self, job_id: str, filename: str) -> Path:
        suffix = Path(filename).suffix[:16]
        return self.uploads_dir / f"{job_id}{suffix}"

    def submit(self, job_id: str, filename: str, language: str | None, upload_path: Path) -> None:
        self.store.create(job_id, filename, language, upload_path)
        self._pool.submit(self._run, job_id)

    def _run(self, job_id: str) -> None:
        row = self.store.get(job_id)
        if row is None:  # borrado antes de empezar
            return
        upload = Path(row["upload_path"])
        wav_dir: Path | None = None
        self.store.update(job_id, status=PROCESSING, progress=0)
        try:
            wav = extract_audio(upload)
            wav_dir = wav.parent
            last = [0.0]

            def on_progress(done: float, total: float) -> None:
                p = done / total if total else 1.0
                if p - last[0] >= 0.005 or p >= 1.0:  # evita escribir en la BD en cada segmento
                    last[0] = p
                    self.store.update(job_id, progress=min(p, 1.0))

            transcript = self.engine.transcribe(
                wav, language=row["language"] or None, on_progress=on_progress,
            )
            result_path = self.results_dir / f"{job_id}.json"
            result_path.write_text(to_json(transcript), encoding="utf-8")
            self.store.update(job_id, status=DONE, progress=1.0, result_path=str(result_path))
        except Exception as e:  # noqa: BLE001 - se reporta al cliente como error del trabajo
            code = getattr(e, "code", None) or (
                "invalid_media" if type(e).__name__ == "InvalidMediaError" else "internal_error"
            )
            self.store.update(job_id, status=ERROR, error_code=code, error_message=str(e))
        finally:
            if wav_dir:
                shutil.rmtree(wav_dir, ignore_errors=True)
            upload.unlink(missing_ok=True)
            self.store.update(job_id, upload_path=None)

    def load_result(self, row) -> Transcript:
        return Transcript.from_dict(json.loads(Path(row["result_path"]).read_text(encoding="utf-8")))

    def delete(self, job_id: str) -> None:
        row = self.store.get(job_id)
        if row and row["result_path"]:
            Path(row["result_path"]).unlink(missing_ok=True)
        if row and row["upload_path"]:
            Path(row["upload_path"]).unlink(missing_ok=True)
        self.store.delete(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.store.close()
