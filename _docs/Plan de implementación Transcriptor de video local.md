# Plan de implementación: Transcriptor de video local

Oct 29, 2026 · @Gormeno

## Contexto y objetivo

Construir **transcriptor**: una herramienta en Python que recibe un video o audio y devuelve su transcripción, usable desde la terminal (CLI) y como API HTTP. Todo corre en local: ningún archivo sale de la máquina y no se llama a ningún servicio externo de pago.

**Instrucciones para Claude (quien implementa):**

- Implementa las fases en orden. No pases a la siguiente sin cumplir los criterios de aceptación de la actual.
- Al terminar cada fase, ejecuta las pruebas y muéstrame el resultado antes de seguir.
- No uses APIs de transcripción en la nube (OpenAI API, Google Speech, AssemblyAI, Deepgram, etc.). Solo modelos que se descargan y corren en local.
- La descarga inicial de los pesos del modelo desde Hugging Face es aceptable; después todo debe funcionar sin internet.
- Código, comentarios y mensajes al usuario en español. Nombres de variables y funciones en inglés.
- Si una decisión no está cubierta aquí, elige la opción más simple y anótala en `DECISIONES.md`.

**Fuera de alcance (v1):** interfaz web gráfica, autenticación de usuarios, traducción, entrenar un modelo propio.

## Stack tecnológico y requisitos

Python 3.11+ con faster-whisper como motor de transcripción; ffmpeg extrae el audio.

| Componente | Herramienta | Uso |
| --- | --- | --- |
| Lenguaje | Python 3.11+ | Todo el proyecto |
| Extracción de audio | ffmpeg (binario del sistema) | Video → WAV mono 16 kHz |
| Modelo de voz | faster-whisper (Whisper vía CTranslate2) | Transcripción local, CPU o GPU |
| CLI | Typer | Comando `transcriptor` |
| API | FastAPI + Uvicorn | Endpoints HTTP |
| Configuración | pydantic-settings | Variables de entorno y `.env` |
| Pruebas | pytest + httpx | Unitarias e integración |
| Empaquetado | pyproject.toml + Docker | Instalación y despliegue |

**Requisitos de máquina:** 4 GB de RAM mínimo para el modelo `small` en CPU; GPU NVIDIA con CUDA opcional (acelera bastante la transcripción). Modelo por defecto: `small`; configurable a `tiny`, `base`, `medium` o `large-v3`.

## Estructura del proyecto

```
transcriptor/
├── pyproject.toml
├── README.md
├── DECISIONES.md
├── Dockerfile
├── .env.example
├── src/transcriptor/
│   ├── __init__.py
│   ├── config.py        # Settings (modelo, dispositivo, idioma, límites)
│   ├── audio.py         # Extracción de audio con ffmpeg
│   ├── engine.py        # Carga del modelo y transcripción (singleton)
│   ├── formats.py       # Exportar a txt, srt, vtt, json
│   ├── models.py        # Dataclasses / esquemas Pydantic (Segment, Transcript, Job)
│   ├── jobs.py          # Cola de trabajos en segundo plano
│   ├── cli.py           # Comandos Typer
│   └── api.py           # App FastAPI
└── tests/
    ├── fixtures/        # Clip corto de prueba (generado, ver Fase 1)
    ├── test_audio.py
    ├── test_formats.py
    ├── test_engine.py
    ├── test_cli.py
    └── test_api.py
```

Regla: `cli.py` y `api.py` no contienen lógica de transcripción; ambos llaman a `audio.py`, `engine.py` y `formats.py`.

## Fases de implementación

Seis fases en orden; cada una termina con pruebas en verde.

### Fase 1 — Base del proyecto y extracción de audio

- [ ] Crear `pyproject.toml` con dependencias y el script de consola `transcriptor = transcriptor.cli:app`.
- [ ] `config.py`: clase `Settings` con `model_size`, `device` (`auto`/`cpu`/`cuda`), `compute_type`, `default_language`, `max_upload_mb`, `models_dir`. Prefijo de variables de entorno `TRANSCRIPTOR_`.
- [ ] `audio.py`: función `extract_audio(input_path) -> Path` que llama a ffmpeg con `subprocess` (`-vn -ac 1 -ar 16000 -f wav`) en un directorio temporal.
- [ ] Verificar al inicio que `ffmpeg` existe (`shutil.which`) y lanzar un error claro si no.
- [ ] Errores propios: `FFmpegNotFoundError`, `InvalidMediaError` (archivo sin pista de audio o corrupto).
- [ ] Generar el fixture de prueba con ffmpeg (un tono o voz sintética de 5 s) en lugar de subir un archivo binario al repo.

**Acepta si:** `extract_audio` convierte un MP4 y un MP3 a WAV mono 16 kHz, y un archivo de texto renombrado a `.mp4` lanza `InvalidMediaError`.

### Fase 2 — Motor de transcripción

- [ ] `models.py`: `Segment(start: float, end: float, text: str)` y `Transcript(language, duration, segments, text)`.
- [ ] `engine.py`: clase `TranscriptionEngine` que carga `WhisperModel` una sola vez (singleton perezoso) y expone `transcribe(audio_path, language=None, word_timestamps=False) -> Transcript`.
- [ ] Con `language=None`, detectar el idioma automáticamente y devolverlo en el resultado.
- [ ] Activar `vad_filter=True` para saltar silencios.
- [ ] Callback opcional de progreso (segundos procesados / duración total) para usarlo en CLI y API.

**Acepta si:** transcribe un clip real en español corto con texto coherente; la segunda llamada no vuelve a cargar el modelo.

### Fase 3 — Formatos de salida

- [ ] `formats.py`: `to_txt`, `to_srt`, `to_vtt`, `to_json` a partir de un `Transcript`.
- [ ] Tiempos SRT con formato `HH:MM:SS,mmm` y VTT con `HH:MM:SS.mmm`.

**Acepta si:** pruebas unitarias con un `Transcript` construido a mano generan SRT y VTT válidos (numeración, tiempos y saltos de línea correctos).

### Fase 4 — CLI

- [ ] Implementar los comandos descritos en la sección *Especificación de CLI y API*.
- [ ] Barra de progreso en la terminal (Rich, incluido con Typer).
- [ ] Aceptar varios archivos o una carpeta y procesarlos uno tras otro.
- [ ] Códigos de salida: 0 éxito, 1 error de entrada, 2 error interno.

**Acepta si:** `transcriptor transcribe video.mp4 -f srt` deja `video.srt` junto al video y `--help` documenta todas las opciones.

### Fase 5 — API HTTP

- [ ] `api.py` con los endpoints de la especificación.
- [ ] `jobs.py`: cola en memoria con un worker en un hilo (`ThreadPoolExecutor(max_workers=1)`), estados `pending`, `processing`, `done`, `error`.
- [ ] Guardar la subida en un temporal por streaming (sin cargarla entera en memoria) y rechazar archivos sobre `max_upload_mb` con 413.
- [ ] Borrar los archivos temporales al terminar cada trabajo.
- [ ] Cargar el modelo al arrancar la app (evento `lifespan`).

**Acepta si:** subir un video devuelve un `job_id`, consultar el trabajo muestra el progreso y al terminar se descarga el SRT.

### Fase 6 — Empaquetado y documentación

- [ ] `Dockerfile` basado en `python:3.11-slim` con ffmpeg instalado; el modelo se descarga en un volumen (`/models`) para no repetir la descarga.
- [ ] `README.md`: instalación, uso de CLI, uso de API con ejemplos `curl`, tabla de modelos y variables de entorno.
- [ ] `.env.example` con todas las variables.

**Acepta si:** `docker run -v modelos:/models -p 8000:8000 transcriptor` levanta la API y transcribe un archivo de prueba.

## Especificación de CLI y API

### CLI

```
transcriptor transcribe ARCHIVO... [opciones]
  -f, --format   txt | srt | vtt | json   (por defecto: txt; se puede repetir)
  -l, --language es | en | ...            (por defecto: detección automática)
  -m, --model    tiny | base | small | medium | large-v3
  -o, --output   carpeta de salida        (por defecto: junto al archivo)
  --device       auto | cpu | cuda
  --stdout       imprime el resultado en vez de guardarlo

transcriptor models          # lista modelos y cuáles están descargados
transcriptor serve [--host 0.0.0.0] [--port 8000]   # levanta la API
```

### API

| Método | Ruta | Descripción | Respuesta |
| --- | --- | --- | --- |
| GET | `/health` | Estado y modelo cargado | `{"status":"ok","model":"small","device":"cpu"}` |
| POST | `/transcriptions` | Sube archivo (`multipart/form-data`: `file`, `language?`) y crea trabajo | 202 `{"job_id":"…","status":"pending"}` |
| GET | `/transcriptions/{job_id}` | Estado, progreso (0–1) y resultado JSON cuando termina | 200 / 404 |
| GET | `/transcriptions/{job_id}/download?format=srt` | Descarga en txt, srt, vtt o json | archivo / 409 si no ha terminado |
| DELETE | `/transcriptions/{job_id}` | Borra el trabajo y su resultado | 204 |

Errores con forma `{"error":{"code":"invalid_media","message":"…"}}`. La documentación interactiva queda en `/docs` (Swagger, automático en FastAPI).

## Pruebas, configuración y criterios de terminado

**Pruebas.** Las unitarias (formatos, configuración, errores de audio) no cargan el modelo: usan un `TranscriptionEngine` simulado. Las que sí cargan Whisper llevan la marca `@pytest.mark.slow` y usan el modelo `tiny`. Comandos: `pytest -m "not slow"` (rápidas) y `pytest` (todas).

**Variables de entorno:**

| Variable | Por defecto | Descripción |
| --- | --- | --- |
| `TRANSCRIPTOR_MODEL_SIZE` | `small` | Tamaño del modelo Whisper |
| `TRANSCRIPTOR_DEVICE` | `auto` | `cpu`, `cuda` o `auto` |
| `TRANSCRIPTOR_COMPUTE_TYPE` | `int8` | `int8` en CPU, `float16` en GPU |
| `TRANSCRIPTOR_DEFAULT_LANGUAGE` | vacío | Vacío = detección automática |
| `TRANSCRIPTOR_MAX_UPLOAD_MB` | `2048` | Tamaño máximo de subida |
| `TRANSCRIPTOR_MODELS_DIR` | `~/.cache/transcriptor` | Dónde se guardan los pesos |

**Terminado cuando:**

- [ ] Las seis fases cumplen sus criterios de aceptación.
- [ ] `pytest` pasa completo en local.
- [ ] Un video real de 10 minutos se transcribe por CLI y por API, con SRT sincronizado al abrirlo en VLC.
- [ ] Con la red desconectada (modelo ya descargado), la transcripción sigue funcionando.
- [ ] El README permite a otra persona instalarlo y usarlo sin preguntar nada.

**Posibles mejoras (v2):** diarización con pyannote (quién habla), interfaz web simple, cola persistente con SQLite, timestamps por palabra en la API.
