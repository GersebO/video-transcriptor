# Plan: hablantes y resumen 100% local

Oct 29, 2026 · @Gormeno

## Contexto y decisión clave

Todo el procesamiento (hablantes y resumen) corre dentro del programa, en el equipo del usuario, sin servicios externos, sin cuentas, sin tokens y sin otro programa instalado aparte. Solo se descargan archivos de modelo una vez, igual que Whisper; después todo funciona sin internet.

**Punto de partida:** la transcripción ya funciona (fases 1–6: CLI interactiva, API, reuniones largas). Equipo: Windows 11, Python 3.13, Ryzen 7 8845HS (8 núcleos), 32 GB RAM, Radeon 780M, sin GPU NVIDIA.

**Qué significa "sin APIs externas" en este plan:**

- No se usa la API de Claude, OpenAI ni ningún servicio en la nube.
- No se usa Ollama ni otro servidor aparte: el modelo de lenguaje se carga **dentro** del propio programa con `llama-cpp-python`, igual que hoy se carga Whisper con `faster-whisper`.
- No se entrena un modelo de lenguaje propio: requiere millones de dólares en cómputo. Se usa un modelo abierto descargable (archivo GGUF) y el programa lo ejecuta localmente.
- Además se incluye un modo de resumen **sin IA** (Fase 9), programado desde cero, por si se prefiere no usar ningún modelo de lenguaje.

**Instrucciones para Claude (quien implementa):**

- Implementa las fases 7, 8 y 9 en orden; no avances sin pruebas en verde y sin mostrarme el resultado.
- Prohibido: llamadas HTTP a servicios de IA, Ollama, pyannote.audio con token, cualquier dependencia que exija cuenta o token.
- Antes de instalar algo pesado (modelos, herramientas de compilación), avísame el tamaño y pide aprobación.
- Verifica en PyPI y Hugging Face las versiones y nombres de modelos vigentes antes de fijarlos; los de este plan son referencias.
- Hablantes y resumen deben poder ejecutarse **sobre una transcripción ya hecha**, sin volver a transcribir horas de audio.
- Registra cada decisión no cubierta aquí en `DECISIONES.md`.

## Cómo encaja todo

Cada etapa guarda su resultado en disco, así se puede repetir una sin rehacer las anteriores.

1. **Audio:** video → `reunion.wav` (ya existe). Se conserva el WAV mientras falten etapas que lo necesiten.
2. **Transcripción:** WAV → `reunion.json` con segmentos y tiempos (ya existe).
3. **Hablantes (Fase 7):** WAV → turnos de habla → se asignan a los segmentos → `reunion.json` actualizado con `speaker`, más `reunion.speakers.json` con los nombres.
4. **Resumen (Fase 8 o 9):** `reunion.json` → `reunion.resumen.md`. Usa los nombres reales si existen.
5. **Exportación:** `.txt`, `.srt`, `.vtt` se regeneran desde el JSON con los hablantes y nombres actuales.

Como la transcripción de la reunión con Gilda ya está corriendo sin hablantes, la Fase 7 debe poder aplicarse después sobre su JSON y su video.

## Fase 7 — Identificar quién habla

Se separan las voces con sherpa-onnx (modelos ONNX de segmentación y de huella de voz, en CPU, sin PyTorch ni token) y luego el usuario pone los nombres.

**Tareas**

- [ ] Prueba de viabilidad primero: instalar `sherpa-onnx` en Python 3.13 / Windows, descargar los modelos y diarizar un audio de 2 voces sintéticas (voces SAPI de Windows). Si no funciona, **detenerse y avisar**; no cambiar a pyannote ni a nada con token.
- [ ] `diarization.py`: `diarize(wav_path, num_speakers=None) -> list[Turn]` con `Turn(start, end, speaker_id)`. Procesar el audio por bloques para que la memoria no crezca en reuniones de horas; callback de progreso.
- [ ] `assign_speakers(transcript, turns)`: a cada segmento se le asigna el hablante con **mayor solapamiento de tiempo**; si un segmento cruza dos hablantes casi por igual, dividirlo por palabras usando `word_timestamps` cuando existan.
- [ ] Unir turnos muy cortos (< 0,5 s) con el vecino para evitar saltos falsos de hablante.
- [ ] `models.py`: `Segment.speaker: str | None`; `Transcript.from_dict` tolera JSON antiguos sin `speaker`.
- [ ] **Nombres**: tras diarizar, por cada "Persona N" mostrar 3 frases de ejemplo con su tiempo y ofrecer reproducir 5 s de su voz (`ffplay -nodisp -autoexit -ss T -t 5`). El usuario escribe el nombre o Enter para dejar "Persona N". Se guarda en `reunion.speakers.json`.
- [ ] Comando `transcriptor hablantes VIDEO --transcript reunion.json` para aplicar la fase a una transcripción ya existente (reextrae el audio si hace falta).
- [ ] Comando `transcriptor nombres reunion.json` para cambiar nombres después sin volver a diarizar.
- [ ] `formats.py`: txt agrupa líneas seguidas del mismo hablante (`Gilda: …`); srt `[Gilda] …`; vtt `<v Gilda>…`; json incluye `speaker`. Sin hablantes, la salida queda idéntica a la actual.

**Acepta si:** en el audio sintético salen 2 hablantes bien asignados en más del 90 % de los segmentos; en la reunión real el usuario revisa 10 frases al azar y está de acuerdo; renombrar con `transcriptor nombres` regenera txt y srt con los nombres nuevos.

**Limitación a documentar:** voces muy parecidas, gente hablando a la vez o un solo micrófono lejano pueden confundir hablantes; indicar `--num-speakers` mejora el resultado.

## Fase 8 — Resumen con IA local embebida

Un modelo de lenguaje abierto (archivo GGUF) se ejecuta dentro del programa con `llama-cpp-python`. Las reuniones largas se resumen por tramos y luego se unen los resultados.

**Modelo**

| Opción | Tamaño aprox. | Uso |
| --- | --- | --- |
| Qwen3-8B Instruct, cuantizado Q4\_K\_M | \~5 GB | Por defecto: buen español, cabe holgado en 32 GB |
| Qwen3-4B Instruct, Q4\_K\_M | \~2,5 GB | Más rápido, algo menos preciso |

Nombre del archivo y repositorio configurables (`TRANSCRIPTOR_LLM_MODEL`); Claude verifica en Hugging Face el modelo vigente antes de fijarlo. Descarga única a `models_dir` con barra de progreso y aviso de tamaño.

**Instalación en Windows**

- [ ] Intentar primero la rueda precompilada de CPU de `llama-cpp-python`. Si no hay para Python 3.13, proponer: compilar con Visual Studio Build Tools (pedir aprobación) o un venv con Python 3.12 solo si es necesario.
- [ ] Opcional, después de que funcione en CPU: compilar con Vulkan para usar la Radeon 780M y medir si mejora.
- [ ] `n_threads=8` (núcleos físicos), `temperature` baja (0,2).

**Cómo resume (`summary.py`)**

1. **Dividir:** cortar la transcripción en tramos de \~3.000 tokens, siempre en un cambio de hablante o pausa, con tiempos `[HH:MM:SS]` y nombres en cada línea.
2. **Extraer por tramo (map):** el modelo devuelve JSON con salida forzada por esquema (gramática de llama.cpp): `temas`, `decisiones`, `tareas` (qué, responsable, plazo si se menciona, minuto), `preguntas_abiertas`, `datos_importantes`. Instrucción: usar solo lo dicho en el texto; si no se sabe el responsable, `null`.
3. **Guardar cada tramo** en `reunion.resumen.partial.jsonl`; si se corta, al reanudar se saltan los tramos ya hechos.
4. **Unir (reduce):** combinar y quitar duplicados en código (no con el modelo); luego una última llamada redacta el resumen general en 1–2 párrafos a partir de las notas ya unidas.
5. **Escribir** `reunion.resumen.md`.

**Formato de `reunion.resumen.md`**

```markdown
# Resumen: Reunión con Gilda (29-10-2026, 1 h 42 min)
Participantes: Germán, Gilda

## Resumen general
…
## Temas tratados
- … [00:12:30]
## Decisiones
- … [00:45:10]
## Tareas pendientes
| Tarea | Responsable | Plazo | Minuto |
## Preguntas abiertas
- …
```

Cada punto lleva el minuto donde se dijo, para poder verificarlo en el SRT o el video.

**Acepta si:** el resumen de la reunión real se genera sin internet; el usuario revisa 5 decisiones o tareas contra el minuto citado y son correctas; cortar el proceso a la mitad y reanudar no rehace los tramos terminados. Medir y anotar en el README cuánto tarda por hora de reunión.

## Fase 9 — Resumen sin IA (extractivo)

Un segundo modo que no usa ningún modelo de lenguaje: selecciona las frases más importantes de la reunión con algoritmos clásicos, todo programado en el proyecto. Es instantáneo, pero no redacta ni deduce tareas: solo elige frases textuales.

- [ ] `extractive.py`, solo con Python + numpy (sin scikit-learn ni NLTK):
  - Unir segmentos en oraciones; limpiar muletillas ("eh", "este", "o sea") y repeticiones.
  - Lista propia de palabras vacías en español (stopwords) dentro del proyecto.
  - Puntuar cada oración con TF-IDF y **TextRank** (grafo de similitud coseno entre oraciones + PageRank iterativo).
  - Elegir las \~10 mejores sin repetir ideas (MMR: penalizar las muy parecidas a las ya elegidas) y mostrarlas en orden cronológico con hablante y minuto.
- [ ] Detección por reglas de **posibles** tareas y decisiones: frases con "hay que", "vamos a", "quedamos en", "me encargo", "para el lunes", etc. Se marcan como "posibles" para revisión humana.
- [ ] Palabras clave de la reunión (top 15 por TF-IDF).
- [ ] Mismo archivo de salida `reunion.resumen.md`, indicando arriba "Resumen extractivo (sin IA)".

**Acepta si:** pruebas unitarias de TextRank y MMR con textos pequeños conocidos; sobre la reunión real tarda menos de 10 s y el usuario considera útiles las frases elegidas.

## Cambios en CLI y API

### CLI

```
transcriptor transcribe VIDEO [--hablantes/--sin-hablantes] [--num-speakers N]
                              [--resumen ia|extractivo|no]
transcriptor hablantes VIDEO --transcript reunion.json [--num-speakers N]
transcriptor nombres   reunion.json                 # renombrar Persona N
transcriptor resumir   reunion.json [--modo ia|extractivo] [--modelo ARCHIVO.gguf]
transcriptor models                                  # incluye modelos de voz y de resumen
```

Modo interactivo, en este orden: "¿Identificar quién habla? \[S/n\]" → paso de nombres → "¿Generar resumen? \[ia/extractivo/no\]".

### API

| Método | Ruta | Cambio |
| --- | --- | --- |
| POST | `/transcriptions` | Nuevos campos `diarize` (bool), `num_speakers` (int), `summary` (`ia`, `extractivo`, `no`) |
| PUT | `/transcriptions/{id}/speakers` | Cuerpo `{"Persona 1":"Gilda"}`; regenera salidas |
| POST | `/transcriptions/{id}/summary` | Lanza el resumen como trabajo aparte (modo en el cuerpo) |
| GET | `/transcriptions/{id}/summary` | Devuelve el Markdown, o 409 si no está listo |

El resumen con IA corre en la misma cola SQLite, con un worker; nunca en paralelo con una transcripción para no saturar la CPU.

## Pruebas y criterios de terminado

**Pruebas automáticas**

- `assign_speakers` con turnos hechos a mano (solapamientos, segmentos que cruzan dos hablantes, turnos cortos).
- Formatos con y sin hablantes; JSON antiguo sin `speaker` sigue cargando.
- `summary.py` con un modelo simulado que devuelve JSON fijo: división en tramos, reanudación, unión y quitado de duplicados.
- TextRank, MMR y reglas de tareas con textos pequeños conocidos.
- Pruebas lentas (`@pytest.mark.slow`): diarización con 2 voces SAPI y resumen con el modelo pequeño (Qwen3-4B) sobre una transcripción de 5 minutos.

**Terminado cuando:**

- [ ] `pytest` completo en verde.
- [ ] Sobre la reunión con Gilda, sin volver a transcribir: `.txt` y `.srt` muestran nombres reales.
- [ ] `reunion.resumen.md` generado en modo IA y en modo extractivo, revisado con el usuario.
- [ ] Todo funciona con el WiFi desconectado y `HF_HUB_OFFLINE=1` (modelos ya descargados).
- [ ] README con tiempos medidos en este equipo: diarización y resumen por hora de reunión, y espacio en disco de cada modelo.
- [ ] `DECISIONES.md` actualizado.
