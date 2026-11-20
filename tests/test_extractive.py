import time

import numpy as np

from transcriptor import extractive as ex
from transcriptor.models import Segment, Transcript


def test_textrank_la_oracion_central_gana_y_suma_uno():
    sim = np.array([
        [0.0, 0.9, 0.8, 0.05],
        [0.9, 0.0, 0.7, 0.05],
        [0.8, 0.7, 0.0, 0.05],
        [0.05, 0.05, 0.05, 0.0],  # frase aislada
    ])
    rank = ex.textrank(sim)
    assert abs(rank.sum() - 1.0) < 1e-6
    assert rank.argmax() == 0 and rank.argmin() == 3


def test_textrank_sin_similitudes_reparte_por_igual():
    rank = ex.textrank(np.zeros((3, 3)))
    assert np.allclose(rank, 1 / 3)
    assert ex.textrank(np.zeros((0, 0))).size == 0


def test_mmr_evita_elegir_dos_frases_casi_iguales():
    scores = np.array([1.0, 0.95, 0.9, 0.1])
    sim = np.full((4, 4), 0.1)
    sim[0, 1] = sim[1, 0] = 0.99  # las frases 0 y 1 dicen casi lo mismo
    np.fill_diagonal(sim, 1.0)
    assert ex.mmr_select(scores, sim, 2) == [0, 2]  # la 1 se descarta por repetir a la 0
    assert sorted(ex.mmr_select(scores, sim, 10)) == [0, 1, 2, 3]  # k mayor que N: devuelve todas


def test_tfidf_filas_normalizadas_y_palabra_rara_pesa_mas():
    x, vocab, idf = ex.tfidf_matrix([["proyecto", "plazo"], ["proyecto", "presupuesto"], ["proyecto"]])
    assert np.allclose(np.linalg.norm(x, axis=1), 1.0)
    assert idf[vocab.index("presupuesto")] > idf[vocab.index("proyecto")]


def test_limpia_muletillas_y_repeticiones_sin_tocar_este_proyecto():
    assert ex.clean_text("Eh, o sea, que que que vamos a revisar este proyecto, ¿no?") == "que vamos a revisar este proyecto"
    assert ex.clean_text("Este, vamos a empezar") == "vamos a empezar"


def test_tokens_sin_vacias_ni_acentos():
    assert ex.tokens("La reunión de planificación es MUY importante") == ["reunion", "planificacion", "importante"]


def test_oraciones_con_tiempo_y_hablante():
    t = Transcript("es", 20, [
        Segment(0, 3, "Hola a todos. Hoy vamos a ver", speaker="Persona 1"),
        Segment(3, 6, "el presupuesto del proyecto.", speaker="Persona 1"),
        Segment(6, 9, "De acuerdo, empecemos", speaker="Persona 2"),
    ], speakers={"Persona 2": "Gilda"})
    got = [(s.start, s.speaker, s.text) for s in ex.split_sentences(t)]
    assert got == [
        (0, "Persona 1", "Hola a todos."),
        (0, "Persona 1", "Hoy vamos a ver el presupuesto del proyecto."),
        (6, "Gilda", "De acuerdo, empecemos"),
    ]


def test_reglas_de_tareas_y_decisiones():
    s = [ex.Sentence(0, None, x) for x in (
        "Me encargo de enviar el informe para el viernes.",
        "Entonces decidimos usar la opción dos.",
        "Ayer hablamos del clima.",
        "Hay que revisar el presupuesto antes del lunes.",
    )]
    assert [x.text for x in ex.find_candidates(s, ex.TASK_PATTERNS)] == [s[0].text, s[3].text]
    assert [x.text for x in ex.find_candidates(s, ex.DECISION_PATTERNS)] == [s[1].text]


def _reunion(n_temas=60):
    temas = [
        "el presupuesto del proyecto de software necesita aprobación del directorio esta semana",
        "la migración de la base de datos a la nube reduce los costos de infraestructura mensuales",
        "el equipo de ventas reporta clientes nuevos interesados en la plataforma de facturación",
    ]
    segs = []
    for i in range(n_temas):
        segs.append(Segment(i * 12.0, i * 12.0 + 10, f"Entonces {temas[i % 3]} número {i}.", speaker="Persona 1" if i % 2 else "Persona 2"))
    segs.append(Segment(n_temas * 12.0, n_temas * 12.0 + 5, "Me encargo de enviar la propuesta para el viernes."))
    return Transcript("es", (n_temas + 1) * 12.0, segs, speakers={"Persona 1": "Germán"})


def test_resumen_completo_markdown_y_rapido(tmp_path):
    t = _reunion()
    t0 = time.time()
    out = ex.summarize_extractive(t, tmp_path / "r.resumen.md", title="Prueba")
    assert time.time() - t0 < 10
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# Resumen: Prueba (12 min)\n_Resumen extractivo (sin IA)")
    assert "Participantes: Persona 2, Germán" in md  # en orden de aparición
    frases = md.split("## Frases más importantes\n")[1].split("\n\n")[0].splitlines()
    assert 1 <= len(frases) <= ex.TOP_SENTENCES and all(f.startswith("- [") for f in frases)
    stamps = [f[3:11] for f in frases]
    assert stamps == sorted(stamps)  # orden cronológico
    assert "Me encargo de enviar la propuesta para el viernes." in md.split("## Posibles tareas")[1]
    claves = md.split("## Palabras clave\n")[1].strip().split(", ")
    assert len(claves) == ex.TOP_KEYWORDS and "migracion" in claves and "para" not in claves


def test_sin_texto_no_falla(tmp_path):
    out = ex.summarize_extractive(Transcript("es", 5, [Segment(0, 1, "ok")]), tmp_path / "r.md")
    md = out.read_text(encoding="utf-8")
    assert "_No hay texto suficiente._" in md and "_Ninguna detectada._" in md


def test_una_hora_de_reunion_tarda_menos_de_10_segundos(tmp_path):
    t0 = time.time()
    ex.summarize_extractive(_reunion(n_temas=300), tmp_path / "r.md")
    assert time.time() - t0 < 10
