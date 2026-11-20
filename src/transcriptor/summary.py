"""Resumen de reuniones con un modelo de lenguaje local (GGUF vía llama-cpp-python).

Las reuniones largas se resumen por tramos (map), se unen en código sin duplicados (reduce) y
una última llamada redacta el resumen general. Cada tramo se guarda en disco para poder reanudar."""

import copy
import difflib
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

from .config import Settings, get_settings
from .models import Transcript

CHUNK_TOKENS = 3000
CHARS_PER_TOKEN = 3.3  # estimación para español; evita cargar el tokenizador solo para contar
PAUSE_SECONDS = 1.5  # pausa que permite cortar un tramo aunque no cambie el hablante
MIN_CHARS_FOR_TOPIC = 300  # por debajo de esto un tramo puede no tener ningún tema
SIMILARITY = 0.85  # dos puntos así de parecidos se consideran el mismo

ProgressCallback = Callable[[int, int], None]  # (tramos hechos, tramos totales)

SECTIONS = ("temas", "decisiones", "preguntas_abiertas", "datos_importantes")

_ITEM = {
    "type": "object",
    "properties": {"texto": {"type": "string"}, "minuto": {"type": "string"}},
    "required": ["texto", "minuto"],
}
_TASK = {
    "type": "object",
    "properties": {
        "que": {"type": "string"},
        "responsable": {"type": ["string", "null"]},
        "plazo": {"type": ["string", "null"]},
        "minuto": {"type": "string"},
    },
    "required": ["que", "responsable", "plazo", "minuto"],
}
CHUNK_SCHEMA = {
    "type": "object",
    "properties": {
        "temas": {"type": "array", "items": _ITEM},
        "decisiones": {"type": "array", "items": _ITEM},
        "tareas": {"type": "array", "items": _TASK},
        "preguntas_abiertas": {"type": "array", "items": _ITEM},
        "datos_importantes": {"type": "array", "items": _ITEM},
    },
    "required": ["temas", "decisiones", "tareas", "preguntas_abiertas", "datos_importantes"],
}
OVERVIEW_SCHEMA = {
    "type": "object",
    "properties": {"resumen": {"type": "string"}},
    "required": ["resumen"],
}

CHUNK_PROMPT = """Eres un asistente que toma notas de reuniones. Lee este fragmento de una transcripción \
(cada línea empieza con [HH:MM:SS] y el nombre de quien habla) y extrae:
- temas: de qué se habló.
- decisiones: lo que se acordó o decidió.
- tareas: cosas que alguien debe hacer (que, responsable, plazo).
- preguntas_abiertas: dudas que quedaron sin resolver.
- datos_importantes: cifras, fechas, nombres, enlaces o datos concretos mencionados.

Reglas:
- Usa SOLO lo que dice el texto. No inventes nada. Si no hay nada para una lista, déjala vacía.
- En "responsable" y "plazo" pon null si el texto no los dice.
- "minuto" es la marca [HH:MM:SS] de la línea donde se dijo, escrita como HH:MM:SS.
- Escribe en español, con frases cortas y claras.

Fragmento:
{text}"""

OVERVIEW_PROMPT = """Con estas notas de una reunión, escribe un resumen general de 2 a 4 oraciones, en español. \
Usa SOLO la información de las notas: no agregues ventajas, riesgos, conclusiones ni explicaciones que no estén en ellas.

Notas:
{notes}"""


class LLM(Protocol):
    """Lo mínimo que necesita este módulo de un modelo de lenguaje (se sustituye por uno falso en tests)."""

    def complete(self, prompt: str, schema: dict, max_tokens: int) -> str: ...


class LlamaLLM:
    """Modelo GGUF ejecutado dentro del programa, con salida JSON forzada por gramática."""

    def __init__(self, settings: Settings | None = None, model_path: str | Path | None = None):
        from llama_cpp import Llama  # import perezoso: es pesado

        self.settings = settings or get_settings()
        self._llm = Llama(
            model_path=str(model_path or model_file(self.settings)),
            n_ctx=self.settings.llm_context,
            n_threads=self.settings.llm_threads,
            verbose=False,
        )

    def complete(self, prompt: str, schema: dict, max_tokens: int) -> str:
        out = self._llm.create_chat_completion(
            # "/no_think" desactiva el razonamiento largo de Qwen3: aquí solo interesa el JSON
            messages=[{"role": "user", "content": prompt + "\n\n/no_think"}],
            response_format={"type": "json_object", "schema": schema},
            temperature=self.settings.llm_temperature,
            max_tokens=max_tokens,
        )
        return out["choices"][0]["message"]["content"]


def llm_dir(settings: Settings | None = None) -> Path:
    return (settings or get_settings()).models_dir / "llm"


def model_file(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    return llm_dir(settings) / settings.llm_model


def model_ready(settings: Settings | None = None) -> bool:
    return model_file(settings).exists()


def ensure_model(settings: Settings | None = None, on_progress: Callable[[int, int], None] | None = None) -> Path:
    """Descarga una sola vez el modelo GGUF (varios GB). Reanuda si quedó a medias."""
    settings = settings or get_settings()
    dest = model_file(settings)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = f"https://huggingface.co/{settings.llm_repo}/resolve/main/{settings.llm_model}"
    done = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={done}-"} if done else {})
    with urllib.request.urlopen(req) as r:  # noqa: S310 (https fijo)
        if done and r.status != 206:  # el servidor ignoró el Range: empezar de cero
            done = 0
        total = done + int(r.headers.get("Content-Length") or 0)
        with open(part, "ab" if done else "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)
    part.replace(dest)
    return dest


# --- utilidades de texto ---

def clock(seconds: float) -> str:
    h, rest = divmod(int(seconds), 3600)
    m, s = divmod(rest, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def estimate_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _seconds(minute: str) -> int:
    """'HH:MM:SS' o 'MM:SS' a segundos; 0 si no se entiende."""
    parts = re.findall(r"\d+", str(minute))
    if not 2 <= len(parts) <= 3:
        return 0
    nums = [int(p) for p in parts]
    while len(nums) < 3:
        nums.insert(0, 0)
    return nums[0] * 3600 + nums[1] * 60 + nums[2]


# --- división en tramos ---

def transcript_lines(t: Transcript) -> list[tuple[float, float, str | None, str]]:
    """(inicio, fin, hablante, texto) de cada segmento con texto."""
    return [(s.start, s.end, t.label(s.speaker), s.text.strip()) for s in t.segments if s.text.strip()]


def format_line(start: float, who: str | None, text: str) -> str:
    return f"[{clock(start)}] {who}: {text}" if who else f"[{clock(start)}] {text}"


def split_chunks(t: Transcript, max_tokens: int = CHUNK_TOKENS) -> list[str]:
    """Tramos de ~max_tokens. Se corta al pasar del 80 % del tamaño en un cambio de hablante o una
    pausa; si no aparece ninguno, se corta de todos modos al llegar a 1,3 veces el tamaño."""
    chunks: list[str] = []
    current: list[str] = []
    used = 0
    prev_end, prev_who = None, None
    for start, end, who, text in transcript_lines(t):
        line = format_line(start, who, text)
        cost = estimate_tokens(line)
        natural = prev_end is not None and (who != prev_who or start - prev_end >= PAUSE_SECONDS)
        if current and ((used >= 0.8 * max_tokens and natural) or used + cost > 1.3 * max_tokens):
            chunks.append("\n".join(current))
            current, used = [], 0
        current.append(line)
        used += cost
        prev_end, prev_who = end, who
    if current:
        chunks.append("\n".join(current))
    return chunks


# --- map: un tramo -> notas ---

_NULLS = {"", "null", "none", "n/a", "no", "ninguno", "ninguna", "no se menciona", "no especificado", "-", "—"}


def _clean_value(value):
    """El modelo a veces escribe "null" como texto: se normaliza a None."""
    if isinstance(value, str) and value.strip().lower().rstrip(".") in _NULLS:
        return None
    return value.strip() if isinstance(value, str) else value


def _parse(raw: str) -> dict:
    data = json.loads(raw)
    out = {k: [] for k in (*SECTIONS, "tareas")}
    for key in out:
        for item in data.get(key) or []:
            if isinstance(item, dict):
                out[key].append({k: _clean_value(v) for k, v in item.items()})
    return out


def _schema_for(text: str) -> dict:
    """Con texto de verdad, el modelo tiende a devolver todo vacío si se le deja: se exige un tema.
    En un tramo casi vacío ("ok", "gracias") no se exige, para no forzar que invente."""
    if len(text) < MIN_CHARS_FOR_TOPIC:
        return CHUNK_SCHEMA
    schema = copy.deepcopy(CHUNK_SCHEMA)
    schema["properties"]["temas"]["minItems"] = 1
    return schema


def summarize_chunk(llm: LLM, text: str) -> dict:
    for attempt in range(2):  # un reintento por si el JSON sale truncado
        try:
            return _parse(llm.complete(CHUNK_PROMPT.format(text=text), _schema_for(text), max_tokens=1500))
        except (ValueError, AttributeError):
            if attempt:
                raise
    raise AssertionError("inalcanzable")


def _chunk_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def partial_path(output: str | Path) -> Path:
    """reunion.resumen.md -> reunion.resumen.partial.jsonl"""
    p = Path(output)
    return p.with_name(f"{p.stem}.partial.jsonl")


def _load_partial(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                done[row["id"]] = row["notes"]
            except (ValueError, KeyError):
                continue  # línea cortada por una interrupción
    except OSError:
        pass
    return done


def map_chunks(
    llm: LLM, chunks: list[str], partial: Path | None = None, on_progress: ProgressCallback | None = None
) -> list[dict]:
    """Notas de cada tramo. Los tramos ya guardados en `partial` no se vuelven a procesar."""
    done = _load_partial(partial) if partial else {}
    notes: list[dict] = []
    out = open(partial, "a", encoding="utf-8") if partial else None
    try:
        for i, text in enumerate(chunks):
            cid = _chunk_id(text)
            if cid not in done:
                done[cid] = summarize_chunk(llm, text)
                if out:
                    out.write(json.dumps({"id": cid, "notes": done[cid]}, ensure_ascii=False) + "\n")
                    out.flush()
            notes.append(done[cid])
            if on_progress:
                on_progress(i + 1, len(chunks))
    finally:
        if out:
            out.close()
    return notes


# --- reduce: unir sin duplicados (en código, no con el modelo) ---

def _main_text(item: dict) -> str:
    return str(item.get("texto") or item.get("que") or "").strip()


def merge_notes(all_notes: list[dict]) -> dict:
    """Junta las notas de todos los tramos, quita las repetidas y ordena por minuto."""
    merged: dict[str, list[dict]] = {k: [] for k in (*SECTIONS, "tareas")}
    for notes in all_notes:
        for key, items in notes.items():
            for item in items:
                text = _normalize(_main_text(item))
                if not text:
                    continue
                for seen in merged[key]:
                    if difflib.SequenceMatcher(None, text, _normalize(_main_text(seen))).ratio() >= SIMILARITY:
                        break  # se queda la primera aparición
                else:
                    merged[key].append(item)
    for items in merged.values():
        items.sort(key=lambda it: _seconds(it.get("minuto", "")))
    return merged


def overview(llm: LLM, merged: dict) -> str:
    lines = []
    for key in (*SECTIONS, "tareas"):
        for item in merged[key]:
            lines.append(f"- ({key}) {_main_text(item)}")
    if not lines:
        return ""
    raw = llm.complete(OVERVIEW_PROMPT.format(notes="\n".join(lines)), OVERVIEW_SCHEMA, max_tokens=700)
    return str(json.loads(raw).get("resumen", "")).strip()


# --- salida ---

def _minute(item: dict) -> str:
    s = _seconds(item.get("minuto", ""))
    return f" [{clock(s)}]" if s or item.get("minuto") else ""


def _cell(value) -> str:
    return str(value).replace("|", "/").strip() if value else "—"


def render_markdown(t: Transcript, merged: dict, general: str, title: str) -> str:
    names = [t.label(s) for s in dict.fromkeys(x.speaker for x in t.segments if x.speaker)]
    head = f"# Resumen: {title} ({_duration(t.duration)})\n"
    if names:
        head += f"Participantes: {', '.join(n for n in names if n)}\n"
    parts = [head.rstrip("\n"), "## Resumen general\n" + (general or "_No hay contenido para resumir._")]

    def bullets(heading: str, key: str) -> None:
        body = "\n".join(f"- {_main_text(i)}{_minute(i)}" for i in merged[key]) or "_Ninguno._"
        parts.append(f"## {heading}\n{body}")

    bullets("Temas tratados", "temas")
    bullets("Decisiones", "decisiones")
    rows = [
        f"| {_cell(i.get('que'))} | {_cell(i.get('responsable'))} | {_cell(i.get('plazo'))} | "
        f"{clock(_seconds(i.get('minuto', '')))} |"
        for i in merged["tareas"]
    ]
    table = "| Tarea | Responsable | Plazo | Minuto |\n| --- | --- | --- | --- |\n" + "\n".join(rows)
    parts.append("## Tareas pendientes\n" + (table if rows else "_Ninguna._"))
    bullets("Preguntas abiertas", "preguntas_abiertas")
    bullets("Datos importantes", "datos_importantes")
    return "\n\n".join(parts) + "\n"


def _duration(seconds: float) -> str:
    h, rest = divmod(int(seconds), 3600)
    m = rest // 60
    return f"{h} h {m:02d} min" if h else f"{m} min"


def summarize(
    t: Transcript,
    llm: LLM,
    output: str | Path,
    title: str = "Reunión",
    on_progress: ProgressCallback | None = None,
) -> Path:
    """Genera `output` (.resumen.md). Reanuda desde el .partial.jsonl si el proceso se cortó."""
    output = Path(output)
    partial = partial_path(output)
    chunks = split_chunks(t)
    merged = merge_notes(map_chunks(llm, chunks, partial, on_progress))
    general = overview(llm, merged)
    output.write_text(render_markdown(t, merged, general, title), encoding="utf-8")
    partial.unlink(missing_ok=True)
    return output
