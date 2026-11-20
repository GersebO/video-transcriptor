# Decisiones

Decisiones no cubiertas por el plan original (se eligió la opción más simple).

- **Sin límite de subida por defecto.** `TRANSCRIPTOR_MAX_UPLOAD_MB=0` significa ilimitado, porque las reuniones son largas. Si es mayor que 0, la API responde 413.
- **Cola en SQLite desde v1** (estaba en v2). Un trabajo de horas debe sobrevivir a un reinicio. Los pendientes se reencolan si su archivo sigue en disco; si no, pasan a `error` (`interrupted`).
- **Un reinicio vuelve a empezar el trabajo** desde cero. No hay reanudación a mitad (`--resume` queda para v2). Los segmentos ya transcritos se vuelcan a un `.partial.jsonl` en la CLI para no perder el texto.
- **`av>=14,<17`.** `av` 17+ rompe `faster-whisper` (`metadata_errors`).
- **`condition_on_previous_text=False`** y VAD con silencios de 500 ms: evitan bucles y alucinaciones en audios largos.
- **Transcripción por bloques de 20 min**, cortados en el punto más silencioso del último minuto. `faster-whisper` decodifica todo el audio a float32 en memoria (medido: 2,4 GB de pico con 3 h), así que sin trocear la RAM crecería con la duración. Con bloques el pico es ~600 MB y no depende de la duración. El idioma detectado en el primer bloque con voz se reutiliza en los demás.
- **Límite práctico de duración: ~35 h de audio.** El WAV se limita a 4 GB y el módulo `wave` de Python no lee RF64. Se acepta porque ninguna reunión se acerca a eso.
- **Comprobación de disco libre** antes de extraer el audio, con un error que sugiere `TRANSCRIPTOR_TEMP_DIR`.
- **Un solo worker** (`max_workers=1`): el modelo ya usa todos los núcleos; paralelizar solo empeora.
- **Resultados en disco** (`results/<job_id>.json`), no en memoria; los formatos se generan al descargar.
- **No se puede borrar un trabajo en proceso** (409); no hay cancelación en v1.
- **Subida en trozos de 1 MB** con `shutil`/lectura en bucle; Starlette ya vuelca a disco las partes grandes del multipart, así que la memoria no crece con el tamaño.
- **Dependencias de la API** (`fastapi`, `uvicorn`) se instalan siempre, por simplicidad.

## Fase 7: hablantes

- **sherpa-onnx 2.0.1 + pyannote-segmentation-3-0 + embedding 3D-Speaker ERes2Net base** (~45 MB en `models_dir/diarization`). Con 2 voces SAPI acertó 6 de 6 segmentos. El embedding está entrenado con chino; si con voces reales confunde hablantes, el siguiente candidato es un WeSpeaker multilingüe.
- **El audio se lee por bloques de 1 min pero se diariza de una sola vez**: sherpa-onnx necesita todo el audio para agrupar las voces; por bloques independientes no se puede garantizar que "Persona 1" sea la misma en cada bloque. El pico de memoria es ~230 MB por hora de audio (float32).
- **`Segment.speaker` guarda un id estable ("Persona N")** y el nombre real vive en `Transcript.speakers` y en `<nombre>.speakers.json`. Así renombrar no toca los segmentos.
- **`Segment.words` y `speaker` se omiten del JSON cuando están vacíos**, para que las transcripciones sin hablantes salgan idénticas a las anteriores.
- **Con hablantes se piden `word_timestamps`** al transcribir, para poder dividir un segmento que cruza dos voces.
- **Un segmento se divide por palabras** solo si el segundo hablante solapa al menos el 60 % que el primero; en otro caso gana el de mayor solape. Un segmento sin solape toma el turno más cercano.
- **Una transcripción `.srt` también sirve de entrada** (`hablantes`, `nombres`): la reunión con Gilda se transcribió antes y solo quedaron `.txt` y `.srt`. El resultado se guarda como `.json` al lado.
- **En modo no interactivo no se preguntan los nombres** (se quedan como "Persona N"); se ponen después con `transcriptor nombres`.
- **Los modelos de voz se descargan solos la primera vez** (aviso de ~45 MB) o con `transcriptor models --descargar-voz`.
- **Umbral de agrupado 0,8 y absorción de hablantes menores**: con 0,5, un video de prueba de 2 min de habla salió con 9 "personas" (una misma voz partida en grupos de 1 a 6 s). Ahora se sube el umbral a 0,8 y los hablantes con menos del 5 % del habla (o menos de 5 s) se absorben en el hablante principal más cercano en el tiempo. Con `--num-speakers` no se absorbe nada. Resultado en esa prueba: 2 hablantes.
- **Nombres vacíos o con BOM se descartan** al cargar (`Transcript.from_dict`, `load_names`, `ask_names`); el BOM aparecía al redirigir entrada desde PowerShell.

## Fase 8: resumen con IA

- **Modelo por defecto Qwen3-4B Q4_K_M (2,5 GB)** y no el 8B del plan: se eligió para empezar más rápido. Se cambia con `TRANSCRIPTOR_LLM_MODEL` / `TRANSCRIPTOR_LLM_REPO` o `--modelo`. `llama-cpp-python` 0.3.36 se instala desde el índice del autor (`abetlen.github.io/llama-cpp-python/whl/cpu`), porque en PyPI no hay rueda para Python 3.13.
- **Se exige al menos un tema por tramo con texto** (`minItems: 1`): sin eso el modelo devolvía todas las listas vacías, que es lo más corto que permite la gramática. En tramos de menos de 300 caracteres no se exige.
- **Los valores "null" escritos como texto se convierten a null** en código (el modelo los escribe como string).
- **Se probó un prompt con más reglas** (no adivinar responsables, qué es una decisión, qué son datos importantes) y fue peor: a veces devolvía todo vacío. Se volvió al prompt simple.
- **Reanudación por contenido**: cada tramo se identifica por un hash de su texto, no por su posición, así que un resumen cortado se reanuda aunque cambien los nombres de hablantes en otros tramos.
- **`--modo extractivo` y la pregunta interactiva "¿Generar resumen?"** quedan para la Fase 9, cuando existan los dos modos. Por ahora existe `resumir` y `transcribe --resumen ia`.

## Fase 9: resumen extractivo

- **Solo Python + numpy** (`extractive.py`): oraciones, limpieza de muletillas, stopwords propias, TF-IDF, TextRank (PageRank iterativo sobre similitud coseno) y MMR (lambda 0,7) para no repetir ideas. Sin scikit-learn ni NLTK.
- **"este" solo se quita como muletilla si va seguido de coma** ("este, vamos a empezar"); "este proyecto" se conserva.
- **Posibles tareas y decisiones por reglas** (frases como "hay que", "me encargo", "vamos a", "decidimos"). Es deliberadamente amplio y marca todo como "revisar": en la reunión de Gilda "vamos a" aparece mucho en demostraciones y da falsos positivos.
- **Tope de 4000 oraciones candidatas** para la matriz de similitud (N x N), que en reuniones de muchas horas crecería demasiado; se quedan las de más contenido.
- **Pregunta interactiva "¿Generar resumen? [ia/extractivo/no]"** con "no" por defecto, después de los nombres.
