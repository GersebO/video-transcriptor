# Transcriptor

Transcribe video y audio **en tu máquina**: ningún archivo sale de ella ni se usan APIs de pago. Usa [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Funciona por terminal (CLI) y como API HTTP. Pensado para **reuniones largas, sin límite de tamaño ni duración**.

> 📖 **Para una guía detallada paso a paso, consulta el [Manual de Uso](MANUAL_DE_USO.md).**
> La documentación de arquitectura y planes previos se encuentra en la carpeta [_docs/](_docs/).

## Guía rápida (Windows)

Ya está instalado. Solo sigue estos pasos.

**Opción A: doble clic.** Abre `transcribir.bat` (en esta carpeta). Te pregunta dónde está el video; pega la ruta y pulsa Enter.

**Opción B: desde PowerShell.**

1. Abre PowerShell y entra a la carpeta del proyecto:
   ```powershell
   cd C:\Users\German\Desktop\German-Proyectos\video-transcriptor
   ```
2. Activa el entorno:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   Si dice que la ejecución de scripts está deshabilitada, ejecuta una vez
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` y repite el paso.
3. Ejecuta:
   ```powershell
   python -m transcriptor
   ```
   (Si tu equipo no bloquea el lanzador, también sirve solo `transcriptor`.)
4. Te pregunta **«¿Dónde está el video, el audio o la carpeta?»**. Pega la ruta, por ejemplo:
   - un video: `C:\Users\German\Videos\Reuniones gilda\Reunion con gilda.mp4`
   - una carpeta entera: `C:\Users\German\Videos\Reuniones gilda` (transcribe todos los videos y audios que haya)

   Puedes arrastrar el archivo a la ventana; las comillas se limpian solas. La próxima vez te propone la última ruta: con Enter la reutilizas.
5. Espera a que termine (hay una barra con el tiempo restante). Los resultados quedan **junto al video**:
   - `Reunion con gilda.txt`: el texto
   - `Reunion con gilda.srt`: subtítulos para VLC

La primera vez descarga el modelo (~485 MB); después funciona sin internet.

**Cambiar idioma, modelo o formato** (opcional): `python -m transcriptor transcribe "ruta" -l es -m medium -f vtt`.
`-m tiny` es el más rápido y `-m medium` el más preciso. Mira más abajo la tabla de modelos.

## Requisitos

- Python 3.11+
- [ffmpeg](https://ffmpeg.org) en el PATH (Windows: `winget install Gyan.FFmpeg`)
- 4 GB de RAM como mínimo para el modelo `small` en CPU. GPU NVIDIA con CUDA opcional.
- La primera vez se descargan los pesos del modelo desde Hugging Face; después funciona sin internet.

## Instalación

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate
pip install -e ".[dev]"
```

## Uso de la CLI

```bash
transcriptor transcribe reunion.mp4                 # deja reunion.txt junto al video
transcriptor transcribe reunion.mp4 -f srt -f txt   # varios formatos
transcriptor transcribe ./reuniones -o ./salida     # una carpeta entera
transcriptor transcribe reunion.mp4 -l es -m medium # idioma y modelo
transcriptor transcribe reunion.mp4 --stdout        # imprime en vez de guardar
transcriptor models                                 # modelos y cuáles están descargados
transcriptor serve --port 8000                      # levanta la API
```

Opciones de `transcribe`: `-f/--format` (txt, srt, vtt, json; repetible), `-l/--language`, `-m/--model`, `-o/--output`, `--device`, `--stdout`.
Códigos de salida: `0` éxito, `1` error de entrada (archivo inválido, formato desconocido), `2` error interno.

## Uso de la API

```bash
transcriptor serve   # http://localhost:8000  (documentación interactiva en /docs)

# 1. Subir (sin límite de tamaño por defecto) -> 202 {"job_id": "...", "status": "pending"}
curl -F "file=@reunion.mp4" -F "language=es" http://localhost:8000/transcriptions

# 2. Consultar estado y progreso (0-1); al terminar incluye el resultado
curl http://localhost:8000/transcriptions/<job_id>

# 3. Descargar en txt, srt, vtt o json (409 si no ha terminado)
curl -o reunion.srt "http://localhost:8000/transcriptions/<job_id>/download?format=srt"

# 4. Borrar el trabajo
curl -X DELETE http://localhost:8000/transcriptions/<job_id>
```

| Método | Ruta | Descripción |
| --- | --- | --- |
| GET | `/health` | Estado y modelo cargado |
| POST | `/transcriptions` | Sube un archivo y crea el trabajo (202) |
| GET | `/transcriptions/{id}` | Estado, progreso y resultado |
| GET | `/transcriptions/{id}/download?format=` | Descarga del resultado |
| DELETE | `/transcriptions/{id}` | Borra el trabajo (204) |

Los errores tienen la forma `{"error":{"code":"...","message":"..."}}`.

## Docker

```bash
docker build -t transcriptor .
docker run -v modelos:/models -v datos:/data -p 8000:8000 transcriptor
```

`/models` guarda los pesos (no se vuelven a descargar) y `/data` la cola de trabajos y las subidas.

## Reuniones largas

- **Sin límite de subida**: `TRANSCRIPTOR_MAX_UPLOAD_MB=0` (por defecto). Ponle un número si quieres un tope (la API responde 413).
- **Memoria acotada**: la subida va a disco en trozos de 1 MB y el audio se transcribe por bloques de 20 min (cortados en silencios). Medido: ~600 MB de RAM con 3 h de audio y el modelo `tiny`, sin depender de la duración.
- **Duración**: hasta ~35 h de audio por archivo (límite del formato WAV).
- **Disco**: el audio temporal ocupa ~115 MB por hora, más el archivo subido. Si tu disco del sistema es pequeño, apunta `TRANSCRIPTOR_TEMP_DIR` a otro. Si falta espacio, se avisa antes de empezar.
- **Tiempo**: en CPU con `small` calcula entre 0,3× y 1× la duración del audio (una reunión de 2 h puede tardar de 40 min a 2 h). Con GPU es mucho más rápido.
- **Reinicios**: la cola está en SQLite. Si el servidor se reinicia, los trabajos pendientes se reencolan y vuelven a empezar. En la CLI, el avance se guarda en un `.partial.jsonl`.
- **Proxy inverso**: si pones nginx delante, usa `client_max_body_size 0;` y `proxy_request_buffering off;`, o recortará las subidas grandes.

## Modelos

| Modelo | Tamaño aprox. | Notas |
| --- | --- | --- |
| `tiny` | ~75 MB | Muy rápido, calidad básica |
| `base` | ~145 MB | Rápido |
| `small` | ~485 MB | **Por defecto**, buen equilibrio en CPU |
| `medium` | ~1,5 GB | Más preciso, lento en CPU |
| `large-v3` | ~3 GB | El mejor; recomendable con GPU |

## Variables de entorno

Prefijo `TRANSCRIPTOR_`; también se leen de un archivo `.env` (ver `.env.example`).

| Variable | Por defecto | Descripción |
| --- | --- | --- |
| `MODEL_SIZE` | `small` | Tamaño del modelo |
| `DEVICE` | `auto` | `cpu`, `cuda` o `auto` |
| `COMPUTE_TYPE` | `int8` | `int8` en CPU, `float16` en GPU |
| `DEFAULT_LANGUAGE` | vacío | Vacío = detección automática |
| `MAX_UPLOAD_MB` | `0` | Máximo de subida; `0` = sin límite |
| `MODELS_DIR` | `~/.cache/transcriptor` | Dónde se guardan los pesos |
| `TEMP_DIR` | vacío | Temporales y subidas (vacío = sistema/junto a la BD) |
| `JOBS_DB` | `~/.cache/transcriptor/jobs.db` | Base de datos de la cola |

## Pruebas

```bash
pytest -m "not slow"   # rápidas, sin cargar el modelo
pytest                 # todas (descargan el modelo tiny)
```

## Hablantes y resumen (todo local)

Funciona sin internet, sin cuentas y sin tokens. Solo se descargan modelos una vez.

```
transcriptor transcribe VIDEO --hablantes [--num-speakers N] [--resumen ia|extractivo|no]
transcriptor hablantes VIDEO --transcript reunion.json   # sobre una transcripción ya hecha (.json o .srt)
transcriptor nombres reunion.json                        # renombrar "Persona N" sin volver a separar voces
transcriptor resumir reunion.json [--modo ia|extractivo] [--modelo ARCHIVO.gguf]
transcriptor models [--descargar-voz]
```

- **Hablantes**: sherpa-onnx (modelos ONNX en CPU). Los nombres se guardan en `reunion.speakers.json`. Voces muy parecidas, gente hablando a la vez o un micrófono lejano pueden confundirse; `--num-speakers` mejora el resultado.
- **Resumen con IA**: modelo GGUF dentro del programa con `llama-cpp-python` (por defecto Qwen3-4B). Por tramos, con reanudación si se corta.
- **Resumen extractivo**: sin ningún modelo; elige frases textuales (TF-IDF + TextRank + MMR) y marca *posibles* tareas y decisiones para revisar.

Espacio en disco: modelos de voz ~45 MB, Qwen3-4B Q4_K_M 2,5 GB (Qwen3-8B Q4_K_M, 5 GB, opcional con `TRANSCRIPTOR_LLM_MODEL`).

Tiempos medidos en este equipo (Ryzen 7 8845HS, CPU):

| Paso | Medida |
| --- | --- |
| Separar voces | 35 s en un video de 20 min (el audio entero cuenta, aunque haya silencio) |
| Resumen con IA (Qwen3-4B) | 83 s para un tramo de ~2 min de habla; el tiempo por hora de reunión real aún no está medido |
| Resumen extractivo | 0,6 s en una reunión de 1 h 10 min |
