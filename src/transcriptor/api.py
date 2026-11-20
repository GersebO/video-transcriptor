"""API HTTP (FastAPI). Sin lógica de transcripción propia: delega en jobs/engine/formats."""

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings, get_settings
from .engine import TranscriptionEngine
from .formats import FORMATS, export
from .jobs import DONE, ERROR, PROCESSING, JobManager

CHUNK = 1024 * 1024  # 1 MB: la subida nunca se carga entera en memoria
MEDIA_TYPES = {
    "txt": "text/plain; charset=utf-8",
    "srt": "application/x-subrip; charset=utf-8",
    "vtt": "text/vtt; charset=utf-8",
    "json": "application/json",
}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


def create_app(settings: Settings | None = None, engine: TranscriptionEngine | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or get_settings()
        eng = engine or TranscriptionEngine.get(s)
        if hasattr(eng, "load"):
            eng.load()  # el modelo se carga al arrancar, no en la primera petición
        app.state.settings = s
        app.state.engine = eng
        app.state.jobs = JobManager(s, eng)
        yield
        app.state.jobs.shutdown()

    app = FastAPI(title="Transcriptor", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return _error(exc.status, exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException):
        return _error(exc.status_code, "not_found" if exc.status_code == 404 else "http_error", str(exc.detail))

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError):
        return _error(422, "invalid_request", "Petición inválida: revisa los campos enviados.")

    def jobs(request: Request) -> JobManager:
        return request.app.state.jobs

    def get_row(request: Request, job_id: str):
        row = jobs(request).store.get(job_id)
        if row is None:
            raise ApiError(404, "job_not_found", f"No existe el trabajo {job_id}.")
        return row

    @app.get("/health")
    def health(request: Request):
        s = request.app.state.settings
        return {
            "status": "ok",
            "model": s.model_size,
            "device": s.device,
            "model_loaded": bool(getattr(request.app.state.engine, "loaded", True)),
        }

    @app.post("/transcriptions", status_code=202)
    def create_transcription(
        request: Request,
        file: UploadFile = File(...),
        language: str | None = Form(None),
    ):
        manager = jobs(request)
        limit_mb = request.app.state.settings.max_upload_mb
        limit = limit_mb * 1024 * 1024 if limit_mb > 0 else None  # 0 = sin límite

        job_id = uuid.uuid4().hex
        filename = Path(file.filename or "archivo").name
        dest = manager.new_upload_path(job_id, filename)
        written = 0
        try:
            with open(dest, "wb") as out:
                while chunk := file.file.read(CHUNK):
                    written += len(chunk)
                    if limit is not None and written > limit:
                        raise ApiError(413, "file_too_large", f"El archivo supera el máximo de {limit_mb} MB.")
                    out.write(chunk)
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
        if written == 0:
            dest.unlink(missing_ok=True)
            raise ApiError(400, "empty_file", "El archivo está vacío.")

        manager.submit(job_id, filename, (language or None), dest)
        return {"job_id": job_id, "status": "pending"}

    @app.get("/transcriptions/{job_id}")
    def get_transcription(request: Request, job_id: str):
        row = get_row(request, job_id)
        body = {
            "job_id": row["id"],
            "status": row["status"],
            "progress": row["progress"],
            "filename": row["filename"],
        }
        if row["status"] == ERROR:
            body["error"] = {"code": row["error_code"], "message": row["error_message"]}
        if row["status"] == DONE:
            import json

            body["result"] = json.loads(export(jobs(request).load_result(row), "json"))
        return body

    @app.get("/transcriptions/{job_id}/download")
    def download(request: Request, job_id: str, format: str = "txt"):
        if format not in FORMATS:
            raise ApiError(400, "invalid_format", f"Formato no soportado: {format}. Usa: {', '.join(FORMATS)}.")
        row = get_row(request, job_id)
        if row["status"] != DONE:
            raise ApiError(409, "not_finished", "El trabajo todavía no ha terminado.")
        content = export(jobs(request).load_result(row), format)
        stem = Path(row["filename"]).stem
        return Response(
            content=content,
            media_type=MEDIA_TYPES[format],
            headers={"Content-Disposition": f'attachment; filename="{stem}.{format}"'},
        )

    @app.delete("/transcriptions/{job_id}", status_code=204)
    def delete_transcription(request: Request, job_id: str):
        row = get_row(request, job_id)
        if row["status"] == PROCESSING:
            raise ApiError(409, "job_running", "El trabajo se está procesando; espera a que termine.")
        jobs(request).delete(job_id)
        return Response(status_code=204)

    return app


app = create_app()
