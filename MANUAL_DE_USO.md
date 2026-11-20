# Manual de Uso - Transcriptor Local de Video y Audio

Transcriptor es una herramienta diseñada para transcribir grabaciones de reuniones, clases, entrevistas o audios de cualquier duración de forma **100% local**:
- **Privacidad total**: Ningún dato ni audio sale de tu máquina.
- **Sin costos**: No usa APIs externas de pago, tokens ni suscripciones.
- **Sin límites**: Procesa archivos de cualquier tamaño o duración (pensado para reuniones largas de varias horas).
- **Funcionalidades avanzadas**: Transcripción a texto/subtítulos, diarización (separar quién habla) y resúmenes automáticos.

---

## Índice

1. [Requisitos Previos](#1-requisitos-previos)
2. [Instalación y Configuración](#2-instalación-y-configuración)
3. [Formas de Uso](#3-formas-de-uso)
   - [Opción 1: Modo Rápido (Doble Clic en Windows)](#opción-1-modo-rápido-doble-clic-en-windows)
   - [Opción 2: Modo Interactivo por Terminal](#opción-2-modo-interactivo-por-terminal)
   - [Opción 3: Modo Comando Directo (CLI)](#opción-3-modo-comando-directo-cli)
   - [Opción 4: Modo Servidor API HTTP](#opción-4-modo-servidor-api-http)
4. [Identificación de Hablantes y Resúmenes](#4-identificación-de-hablantes-y-resúmenes)
   - [Identificar quién habla (Diarización)](#identificar-quién-habla-diarización)
   - [Renombrar participantes](#renombrar-participantes)
   - [Generar resumen de la reunión](#generar-resumen-de-la-reunión)
5. [Modelos Disponibles de Whisper](#5-modelos-disponibles-de-whisper)
6. [Variables de Entorno y Configuración Avanzada](#6-variables-de-entorno-y-configuración-avanzada)
7. [Preguntas Frecuentes y Solución de Problemas](#7-preguntas-frecuentes-y-solución-de-problemas)

---

## 1. Requisitos Previos

1. **Python 3.11 o superior** instalado en el sistema.
2. **FFmpeg** instalado y disponible en la variable PATH del sistema.
   - En Windows puedes instalarlo fácilmente abriendo PowerShell y ejecutando:
     ```powershell
     winget install Gyan.FFmpeg
     ```
   - Para comprobar que funciona: `ffmpeg -version`
3. **Memoria RAM**:
   - Mínimo **4 GB de RAM** para el modelo por defecto (`small`).
   - Si cuentas con tarjeta gráfica NVIDIA (CUDA), la transcripción se acelerará significativamente.

---

## 2. Instalación y Configuración

Si ya clonaste o descargaste el proyecto, inicializa el entorno virtual ejecutando en la raíz del proyecto:

### En Windows (PowerShell):
```powershell
# 1. Crear el entorno virtual si no existe
python -m venv .venv

# 2. Activar el entorno virtual
.\.venv\Scripts\Activate.ps1

# (Si Windows bloquea scripts, ejecuta una sola vez: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# 3. Instalar dependencias en modo editable
pip install -e ".[dev]"
```

### En Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## 3. Formas de Uso

### Opción 1: Modo Rápido (Doble Clic en Windows)

La forma más sencilla sin tocar la terminal:
1. Haz doble clic sobre el archivo **`transcribir.bat`**.
2. Se abrirá una ventana que te preguntará:
   ```text
   ¿Dónde está el video, el audio o la carpeta? (pega la ruta)
   ```
3. **Pega la ruta** del video o **arrastra el archivo directamente a la ventana**.
4. Pulsa `Enter`. El programa te preguntará si deseas identificar hablantes y si quieres generar resumen.
5. Los archivos generados se guardarán automáticamente **en la misma carpeta donde está tu video**.

---

### Opción 2: Modo Interactivo por Terminal

Te guía paso a paso con preguntas en consola:

```powershell
.\.venv\Scripts\Activate.ps1
python -m transcriptor
```

- Si no pasas argumentos, el programa recuerda tu última ruta usada (puedes presionar `Enter` para reutilizarla).
- Te preguntará si deseas detectar hablantes y qué tipo de resumen prefieres (`ia`, `extractivo` o `no`).
- Muestra una barra de progreso en tiempo real con el tiempo estimado restante.

---

### Opción 3: Modo Comando Directo (CLI)

Ideal para automatizaciones, scripts o uso avanzado con parámetros específicos.

#### Transcripción básica:
```bash
transcriptor transcribe "C:\Videos\Reunion.mp4"
```
Genera `Reunion.txt` junto al video.

#### Generar múltiples formatos a la vez:
```bash
transcriptor transcribe "Reunion.mp4" -f txt -f srt -f vtt -f json
```

#### Transcribir una carpeta completa de videos/audios:
```bash
transcriptor transcribe "C:\Videos\Reuniones" -o "C:\Videos\Salida"
```

#### Especificar idioma y modelo:
```bash
# -l: idioma (es, en, fr...) o detección automática si se omite
# -m: modelo (tiny, base, small, medium, large-v3)
transcriptor transcribe "Reunion.mp4" -l es -m medium
```

#### Transcribir con separación de hablantes y resumen en un solo paso:
```bash
transcriptor transcribe "Reunion.mp4" --hablantes --num-speakers 3 --resumen extractivo
```

#### Ver modelos descargados o disponibles:
```bash
transcriptor models
```

---

### Opción 4: Modo Servidor API HTTP

Si deseas integrar la transcripción con otras aplicaciones o usarlo como servicio web local:

1. Inicia el servidor:
   ```bash
   transcriptor serve --port 8000
   ```
2. Abre tu navegador en **`http://localhost:8000/docs`** para ver la documentación interactiva Swagger.

#### Ejemplos vía cURL:
- **Subir archivo para transcripción (asíncrono):**
  ```bash
  curl -F "file=@reunion.mp4" -F "language=es" http://localhost:8000/transcriptions
  # Devuelve: {"job_id": "...", "status": "pending"}
  ```
- **Consultar estado y progreso (0.0 a 1.0):**
  ```bash
  curl http://localhost:8000/transcriptions/<JOB_ID>
  ```
- **Descargar el resultado en el formato deseado:**
  ```bash
  curl -o reunion.srt "http://localhost:8000/transcriptions/<JOB_ID>/download?format=srt"
  ```

---

## 4. Identificación de Hablantes y Resúmenes

El proyecto permite ejecutar la separación de voces y resúmenes **tanto al momento de transcribir como después sobre un archivo ya procesado** (sin tener que volver a transcribir horas de audio).

### Identificar quién habla (Diarización)

Usa modelos ONNX locales de voz (se descargan una sola vez, ~45 MB).

- **Al transcribir:**
  ```bash
  transcriptor transcribe reunion.mp4 --hablantes
  # Si conoces el número exacto de personas, mejora la precisión:
  transcriptor transcribe reunion.mp4 --hablantes --num-speakers 4
  ```

- **Sobre una transcripción previa (.json o .srt):**
  ```bash
  transcriptor hablantes reunion.mp4 --transcript reunion.json
  ```

### Renombrar participantes

Por defecto, los participantes se identifican como "Persona 1", "Persona 2", etc. Puedes asignarles sus nombres reales:

```bash
transcriptor nombres reunion.json --video reunion.mp4
```
El programa te mostrará frases dichas por cada persona (e incluso puede reproducir 5 segundos de audio con `ffplay` para que reconozcas la voz) y guardará los nombres en `reunion.speakers.json`.

### Generar resumen de la reunión

Existen dos modalidades locales:

1. **Modo Extractivo (`extractivo`)**:
   - **Ultrarrápido** (tarda menos de 1 segundo en una reunión de 1 hora).
   - Utiliza algoritmos estadísticos (TF-IDF + TextRank + MMR) en Python/NumPy sin requerir descargar modelos LLM pesados.
   - Extrae oraciones clave y detecta automáticamente posibles tareas y compromisos.
   ```bash
   transcriptor resumir reunion.json --modo extractivo
   ```

2. **Modo Inteligencia Artificial (`ia`)**:
   - Usa un modelo de lenguaje local (GGUF, por defecto Qwen3-4B mediante `llama-cpp-python`).
   - Analiza la reunión por secciones y redacta una minuta estructurada con temas, acuerdos y próximos pasos.
   ```bash
   transcriptor resumir reunion.json --modo ia
   ```

El resultado se genera en un archivo Markdown con formato limpio: `<nombre>.resumen.md`.

---

## 5. Modelos Disponibles de Whisper

| Modelo | Tamaño aprox. | Velocidad | Precisión | Recomendación de uso |
| :--- | :---: | :---: | :---: | :--- |
| `tiny` | ~75 MB | Muy rápida | Básica | Pruebas rápidas o equipos muy limitados. |
| `base` | ~145 MB | Rápida | Media | Audios claros con buen micrófono. |
| `small` | ~485 MB | Equilibrada | Buena | **Recomendado por defecto** (buen desempeño en CPU). |
| `medium` | ~1.5 GB | Moderada | Muy buena | Alta precisión en reuniones formales. |
| `large-v3`| ~3.0 GB | Lenta en CPU | Máxima | Ideal si cuentas con GPU NVIDIA con CUDA. |

> [!NOTE]
> La primera vez que uses un modelo, este se descargará automáticamente desde Hugging Face y quedará guardado en tu caché local (`~/.cache/transcriptor`). A partir de ese momento funcionará completamente sin conexión a internet.

---

## 6. Variables de Entorno y Configuración Avanzada

Puedes personalizar el comportamiento copiando el archivo `.env.example` a `.env` y modificando sus valores:

| Variable | Valor por defecto | Descripción |
| :--- | :--- | :--- |
| `TRANSCRIPTOR_MODEL_SIZE` | `small` | Modelo Whisper predeterminado (`tiny`, `base`, `small`, `medium`, `large-v3`). |
| `TRANSCRIPTOR_DEVICE` | `auto` | Dispositivo de cómputo (`auto`, `cpu`, `cuda`). |
| `TRANSCRIPTOR_COMPUTE_TYPE` | `int8` | Precisión numérica (`int8` para CPU, `float16` para GPU CUDA). |
| `TRANSCRIPTOR_DEFAULT_LANGUAGE`| `None` | Idioma por defecto (ej: `es`, `en`). Si está vacío, detecta automáticamente. |
| `TRANSCRIPTOR_MAX_UPLOAD_MB` | `0` | Límite de subida en la API (en MB). `0` significa sin límite. |
| `TRANSCRIPTOR_MODELS_DIR` | `~/.cache/transcriptor` | Directorio donde se almacenan los pesos de los modelos. |
| `TRANSCRIPTOR_TEMP_DIR` | Sistema | Carpeta para archivos temporales WAV durante la extracción. |

---

## 7. Preguntas Frecuentes y Solución de Problemas

### 1. `FFmpegNotFoundError: No se encontró ffmpeg`
- **Causa**: FFmpeg no está instalado o no se encuentra en las variables de entorno PATH.
- **Solución**: En Windows ejecuta `winget install Gyan.FFmpeg`. Cierra tu consola o reinicia la terminal para que reconozca el nuevo comando.

### 2. Error al activar el entorno en PowerShell (`Activate.ps1 cannot be loaded...`)
- **Causa**: La política de ejecución de scripts de PowerShell de Windows restringe scripts no firmados.
- **Solución**: Abre PowerShell y ejecuta:
  ```powershell
  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  ```
  Luego vuelve a ejecutar `.\.venv\Scripts\Activate.ps1`.

### 3. ¿Cuánto tarda en transcribir una reunión?
- En un procesador moderno estándar (CPU) con el modelo `small`, suele tomar entre **0.3x y 1.0x** la duración del audio. Por ejemplo, una reunión de 1 hora tardará aproximadamente entre 20 y 45 minutos. Si dispones de GPU compatible con CUDA, el tiempo disminuye drásticamente.

### 4. ¿Qué formatos de entrada soporta?
- Prácticamente cualquier formato multimedia decodificable por FFmpeg:
  - **Video**: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.flv`, `.wmv`, `.m4v`, `.ts`
  - **Audio**: `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg`, `.opus`, `.aac`, `.wma`
