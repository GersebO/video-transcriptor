"""Resumen extractivo sin IA: elige las frases más importantes con TF-IDF, TextRank y MMR.

Todo con Python y numpy. No redacta ni deduce: solo selecciona frases textuales, y marca como
"posibles" las que parecen tareas o decisiones para que una persona las revise."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import Transcript
from .summary import clock

TOP_SENTENCES = 10
TOP_KEYWORDS = 15
MIN_CONTENT_WORDS = 5  # frases con menos palabras con contenido no se consideran candidatas
MAX_SENTENCE_CHARS = 320  # sin puntuación del transcriptor, se corta aquí para no tener "frases" enormes
MAX_CANDIDATES = 4000  # limita la matriz de similitud (N x N) en reuniones de muchas horas
DAMPING = 0.85
MMR_LAMBDA = 0.7  # 1 = solo importancia; 0 = solo variedad

# Palabras vacías del español (propias: el proyecto no depende de NLTK ni scikit-learn)
STOPWORDS = frozenset(
    """a al algo algunas algunos ante antes aqui asi aun aunque bien cada casi como con contra cual cuales
    cuando cuanto de del desde donde dos el ella ellas ellos en entre era eran eres es esa esas ese eso esos
    esta estaba estaban estado estamos estan estar estas este esto estos estoy fue fueron ha haber habia han
    hasta hay hace hacer he la las le les lo los mas me mi mis mucho muy nada ni no nos nosotros nuestro o os
    otra otras otro otros para pero poco por porque que quien quienes se sea ser si sido sin sobre sois solo
    son soy su sus tambien tan tanto te tengo tiene tienen todo todos tu tus un una unas uno unos usted va vamos
    van vas ve veces ver vez voy vosotros y ya yo ahi alla ahora entonces luego pues tipo cosa cosas dice digo
    dijo puede pueden podemos puedo ok vale claro bueno buena mira oye verdad igual tal asi""".split()
)

# Muletillas que se quitan antes de puntuar y mostrar
_FILLERS = re.compile(
    r"(?:\b(?:eh+|em+|mm+|mhm|ah+|uh+|o sea|digamos|bueno|pues)\b|\beste(?=,)|¿(?:no|sí|ya|ves|cachai|verdad)\?)"
    r"[,.]?\s*",  # "este" solo cuando va seguido de coma: "este proyecto" no es una muletilla
    re.IGNORECASE,
)
_REPEATED = re.compile(r"\b(\w+)([,.]?\s+\1\b)+", re.IGNORECASE)  # "que que que" -> "que"

# Reglas para marcar posibles tareas y decisiones (el texto se compara sin acentos y en minúsculas)
TASK_PATTERNS = re.compile(
    r"\b(hay que|tenemos que|tengo que|tienes que|tienen que|debemos|deberia(?:mos)?|"
    r"me encargo|te encargas|se encarga|yo lo hago|lo hago yo|lo reviso|lo envio|te envio|te mando|"
    r"quedamos en|queda pendiente|pendiente|para el (?:lunes|martes|miercoles|jueves|viernes|sabado|domingo)|"
    r"para manana|para hoy|antes del|a mas tardar|de aqui a|esta semana|la proxima semana)\b"
)
DECISION_PATTERNS = re.compile(
    r"\b(vamos a|decidimos|decidi|acordamos|quedamos asi|quedamos en|entonces sera|lo dejamos|"
    r"nos quedamos con|se aprobo|aprobado|queda decidido|optamos por|elegimos|definimos)\b"
)


@dataclass
class Sentence:
    start: float
    speaker: str | None
    text: str


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def clean_text(text: str) -> str:
    text = _FILLERS.sub("", text)
    text = _REPEATED.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip(" ,")


def tokens(text: str) -> list[str]:
    """Palabras con contenido: minúsculas, sin acentos, sin vacías ni números sueltos."""
    words = re.findall(r"[a-zñü]{3,}", strip_accents(text.lower()))
    return [w for w in words if w not in STOPWORDS]


def split_sentences(t: Transcript) -> list[Sentence]:
    """Une los segmentos y los corta en oraciones (con el tiempo y hablante de donde empiezan).
    Whisper suele puntuar; si una oración no termina en 320 caracteres, se corta igualmente."""
    out: list[Sentence] = []
    buf, start, who = "", 0.0, None
    for seg in t.segments:
        label = t.label(seg.speaker)
        text = seg.text.strip()
        if not text:
            continue
        if buf and label != who:  # un cambio de hablante siempre cierra la oración
            out.append(Sentence(start, who, buf))
            buf = ""
        if not buf:
            start, who = seg.start, label
        buf = f"{buf} {text}".strip()
        parts = re.split(r"(?<=[.!?…])\s+", buf)
        for done in parts[:-1]:
            out.append(Sentence(start, who, done))
        buf = parts[-1]
        if len(buf) > MAX_SENTENCE_CHARS or re.search(r"[.!?…]$", buf):
            out.append(Sentence(start, who, buf))
            buf = ""
    if buf:
        out.append(Sentence(start, who, buf))
    cleaned = [Sentence(s.start, s.speaker, clean_text(s.text)) for s in out]
    return [s for s in cleaned if s.text]


def tfidf_matrix(docs: list[list[str]]) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Matriz oraciones x vocabulario con filas normalizadas (L2). Devuelve (X, vocabulario, idf)."""
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
    tf = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for r, d in enumerate(docs):
        for w in d:
            tf[r, vocab[w]] += 1
    df = (tf > 0).sum(axis=0)
    idf = np.log((1 + len(docs)) / (1 + df)).astype(np.float32) + 1.0
    x = tf * idf
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms), list(vocab), idf


def textrank(sim: np.ndarray, damping: float = DAMPING, iterations: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank iterativo sobre el grafo de similitud entre oraciones."""
    n = len(sim)
    if n == 0:
        return np.zeros(0)
    w = sim.astype(np.float64).copy()
    np.fill_diagonal(w, 0.0)
    sums = w.sum(axis=1, keepdims=True)
    # Una oración sin ninguna parecida reparte su peso por igual (evita dividir por cero)
    trans = np.where(sums > 0, w / np.where(sums == 0, 1, sums), 1.0 / n)
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        new = (1 - damping) / n + damping * (trans.T @ rank)
        done = np.abs(new - rank).sum() < tol
        rank = new
        if done:
            break
    return rank


def mmr_select(scores: np.ndarray, sim: np.ndarray, k: int, lam: float = MMR_LAMBDA) -> list[int]:
    """Elige k oraciones importantes sin repetir ideas: penaliza las parecidas a las ya elegidas."""
    k = min(k, len(scores))
    if k <= 0:
        return []
    span = scores.max() - scores.min()
    norm = (scores - scores.min()) / span if span > 0 else np.ones_like(scores)
    chosen: list[int] = []
    for _ in range(k):
        best, best_val = -1, -np.inf
        for i in range(len(scores)):
            if i in chosen:
                continue
            redundancy = max((sim[i, j] for j in chosen), default=0.0)
            val = lam * norm[i] - (1 - lam) * redundancy
            if val > best_val:
                best, best_val = i, val
        chosen.append(best)
    return chosen


def keywords(docs: list[list[str]], n: int = TOP_KEYWORDS) -> list[str]:
    """Palabras que más pesan en la reunión: frecuencia total por IDF."""
    if not docs:
        return []
    vocab_tf: dict[str, int] = {}
    df: dict[str, int] = {}
    for d in docs:
        for w in d:
            vocab_tf[w] = vocab_tf.get(w, 0) + 1
        for w in set(d):
            df[w] = df.get(w, 0) + 1
    score = {w: c * (np.log((1 + len(docs)) / (1 + df[w])) + 1.0) for w, c in vocab_tf.items() if c >= 2}
    return [w for w, _ in sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))[:n]]


def find_candidates(sentences: list[Sentence], pattern: re.Pattern) -> list[Sentence]:
    return [s for s in sentences if pattern.search(strip_accents(s.text.lower()))]


def select_sentences(sentences: list[Sentence], k: int = TOP_SENTENCES) -> list[Sentence]:
    """Las k frases más importantes, en orden cronológico."""
    docs = [tokens(s.text) for s in sentences]
    keep = [i for i, d in enumerate(docs) if len(d) >= MIN_CONTENT_WORDS]
    if len(keep) > MAX_CANDIDATES:  # reuniones de muchas horas: quedarse con las de más contenido
        keep = sorted(sorted(keep, key=lambda i: -len(docs[i]))[:MAX_CANDIDATES])
    if not keep:
        return []
    x, _, _ = tfidf_matrix([docs[i] for i in keep])
    sim = x @ x.T
    scores = textrank(sim)
    picked = mmr_select(scores, sim, k)
    return [sentences[keep[i]] for i in sorted(picked, key=lambda p: sentences[keep[p]].start)]


def _line(s: Sentence) -> str:
    who = f"{s.speaker}: " if s.speaker else ""
    return f"- [{clock(s.start)}] {who}{s.text}"


def _duration(seconds: float) -> str:
    h, rest = divmod(int(seconds), 3600)
    return f"{h} h {rest // 60:02d} min" if h else f"{rest // 60} min"


def render_markdown(t: Transcript, title: str, top: list[Sentence], tasks: list[Sentence],
                    decisions: list[Sentence], words: list[str]) -> str:
    names = [t.label(s) for s in dict.fromkeys(x.speaker for x in t.segments if x.speaker)]
    head = f"# Resumen: {title} ({_duration(t.duration)})\n_Resumen extractivo (sin IA): son frases textuales de la reunión._"
    if names:
        head += f"\nParticipantes: {', '.join(n for n in names if n)}"

    def block(heading: str, items: list[Sentence], empty: str) -> str:
        return f"## {heading}\n" + ("\n".join(_line(s) for s in items) if items else f"_{empty}_")

    return "\n\n".join([
        head,
        block("Frases más importantes", top, "No hay texto suficiente."),
        block("Posibles decisiones (revisar)", decisions, "Ninguna detectada."),
        block("Posibles tareas (revisar)", tasks, "Ninguna detectada."),
        "## Palabras clave\n" + (", ".join(words) if words else "_Ninguna._"),
    ]) + "\n"


def summarize_extractive(t: Transcript, output: str | Path, title: str = "Reunión", top: int = TOP_SENTENCES) -> Path:
    """Genera `output` (.resumen.md) sin usar ningún modelo de lenguaje."""
    sentences = split_sentences(t)
    docs = [tokens(s.text) for s in sentences]
    text = render_markdown(
        t, title,
        select_sentences(sentences, top),
        find_candidates(sentences, TASK_PATTERNS),
        find_candidates(sentences, DECISION_PATTERNS),
        keywords(docs),
    )
    output = Path(output)
    output.write_text(text, encoding="utf-8")
    return output
