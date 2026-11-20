"""Configuración del transcriptor (variables de entorno con prefijo TRANSCRIPTOR_)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIPTOR_", env_file=".env", extra="ignore"
    )

    model_size: str = "small"
    device: Literal["auto", "cpu", "cuda"] = "auto"
    compute_type: str = "int8"
    default_language: str = ""  # vacío = detección automática
    # 0 = sin límite (pensado para reuniones largas)
    max_upload_mb: int = 0
    models_dir: Path = Path.home() / ".cache" / "transcriptor"
    # Directorio de temporales; vacío = el del sistema. Útil para apuntar a un disco grande.
    temp_dir: str = ""
    # Resumen con IA local (modelo GGUF que se ejecuta dentro del programa con llama-cpp-python)
    llm_repo: str = "Qwen/Qwen3-4B-GGUF"
    llm_model: str = "Qwen3-4B-Q4_K_M.gguf"
    llm_threads: int = 8  # núcleos físicos
    llm_temperature: float = 0.2
    llm_context: int = 8192
    # Base de datos SQLite de la cola de trabajos de la API
    jobs_db: Path = Path.home() / ".cache" / "transcriptor" / "jobs.db"


@lru_cache
def get_settings() -> Settings:
    return Settings()
