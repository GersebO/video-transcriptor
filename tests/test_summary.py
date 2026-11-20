import json

import pytest

from transcriptor import summary
from transcriptor.models import Segment, Transcript


def make_transcript(n=40):
    segs = []
    for i in range(n):
        who = "Persona 1" if (i // 5) % 2 == 0 else "Persona 2"
        segs.append(Segment(i * 10.0, i * 10.0 + 8, f"Esta es la frase número {i} de la reunión sobre el proyecto", speaker=who))
    return Transcript("es", n * 10.0, segs, speakers={"Persona 1": "Germán", "Persona 2": "Gilda"})


class FakeLLM:
    """Devuelve notas fijas por tramo y cuenta las llamadas."""

    def __init__(self):
        self.calls = 0

    def complete(self, prompt, schema, max_tokens):
        self.calls += 1
        if schema is summary.OVERVIEW_SCHEMA:
            return json.dumps({"resumen": "Se habló del proyecto."})
        return json.dumps(
            {
                "temas": [{"texto": "Avance del proyecto", "minuto": "00:00:10"}],
                "decisiones": [{"texto": "Entregar el viernes", "minuto": "00:01:00"}],
                "tareas": [{"que": "Enviar el informe", "responsable": "Gilda", "plazo": None, "minuto": "00:02:00"}],
                "preguntas_abiertas": [],
                "datos_importantes": [],
            }
        )


def test_tramos_cortan_en_cambio_de_hablante_y_cubren_todo():
    t = make_transcript(200)
    chunks = summary.split_chunks(t, max_tokens=300)
    assert len(chunks) > 3
    joined = "\n".join(chunks)
    assert all(f"frase número {i} " in joined for i in range(200))
    for c in chunks[:-1]:  # los tramos intermedios acaban poco después del 80 % del tamaño
        assert summary.estimate_tokens(c) <= 1.3 * 300 + 40
    assert "Germán:" in chunks[0] and "[00:00:00]" in chunks[0]


def test_tramo_unico_si_es_corto():
    assert len(summary.split_chunks(make_transcript(5))) == 1


def test_une_sin_duplicados_y_ordena_por_minuto():
    a = {"temas": [{"texto": "Presupuesto del proyecto", "minuto": "00:10:00"}], "tareas": [], "decisiones": [],
         "preguntas_abiertas": [], "datos_importantes": []}
    b = {"temas": [{"texto": "presupuesto del proyecto.", "minuto": "00:20:00"},
                   {"texto": "Plazos", "minuto": "00:05:00"}], "tareas": [], "decisiones": [],
         "preguntas_abiertas": [], "datos_importantes": []}
    merged = summary.merge_notes([a, b])
    assert [i["texto"] for i in merged["temas"]] == ["Plazos", "Presupuesto del proyecto"]


def test_resumen_completo_y_formato(tmp_path):
    out = tmp_path / "reunion.resumen.md"
    llm = FakeLLM()
    summary.summarize(make_transcript(10), llm, out, title="Reunión con Gilda")  # hablan los dos
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# Resumen: Reunión con Gilda (1 min)\nParticipantes: Germán, Gilda\n\n## Resumen general")
    assert "Participantes: Germán, Gilda" in md
    assert "## Resumen general\nSe habló del proyecto." in md
    assert "- Entregar el viernes [00:01:00]" in md
    assert "| Enviar el informe | Gilda | — | 00:02:00 |" in md
    assert not summary.partial_path(out).exists()  # se borra al terminar


def test_reanuda_sin_rehacer_los_tramos_terminados(tmp_path):
    t = make_transcript(200)
    out = tmp_path / "r.resumen.md"
    chunks = summary.split_chunks(t)
    assert len(chunks) >= 2

    class SeCorta(FakeLLM):
        def complete(self, prompt, schema, max_tokens):
            if self.calls == 1:  # el segundo tramo falla, como si se cortara el proceso
                raise KeyboardInterrupt
            return super().complete(prompt, schema, max_tokens)

    first = SeCorta()
    with pytest.raises(KeyboardInterrupt):
        summary.summarize(t, first, out)
    partial = summary.partial_path(out)
    assert len(partial.read_text(encoding="utf-8").splitlines()) == 1  # el primer tramo quedó guardado

    second = FakeLLM()
    summary.summarize(t, second, out)
    assert second.calls == (len(chunks) - 1) + 1  # solo los tramos que faltan + el resumen general
    assert out.exists() and not partial.exists()


def test_json_truncado_reintenta_una_vez():
    class Mal(FakeLLM):
        def complete(self, prompt, schema, max_tokens):
            if self.calls == 0:
                self.calls += 1
                return '{"temas": [{"texto": "cortado'
            return super().complete(prompt, schema, max_tokens)

    llm = Mal()
    notes = summary.summarize_chunk(llm, "[00:00:00] hola")
    assert llm.calls == 2 and notes["temas"][0]["texto"] == "Avance del proyecto"


def test_null_escrito_como_texto_se_normaliza():
    class Nulos(FakeLLM):
        def complete(self, prompt, schema, max_tokens):
            return json.dumps({
                "temas": [], "decisiones": [], "preguntas_abiertas": [], "datos_importantes": [],
                "tareas": [{"que": "Subir el archivo", "responsable": "null", "plazo": "No especificado.", "minuto": "00:00:07"}],
            })

    task = summary.summarize_chunk(Nulos(), "[00:00:07] x")["tareas"][0]
    assert task["responsable"] is None and task["plazo"] is None and task["que"] == "Subir el archivo"


def test_exige_un_tema_solo_si_el_tramo_tiene_texto():
    largo, corto = "palabra " * 80, "[00:00:01] Persona 1: ok"
    assert summary._schema_for(largo)["properties"]["temas"]["minItems"] == 1
    assert "minItems" not in summary._schema_for(corto)["properties"]["temas"]
    assert "minItems" not in summary.CHUNK_SCHEMA["properties"]["temas"]  # el original no se modifica


def test_linea_cortada_en_el_parcial_se_ignora(tmp_path):
    p = tmp_path / "x.partial.jsonl"
    p.write_text('{"id": "a", "notes": {"temas": []}}\n{"id": "b", "no', encoding="utf-8")
    assert list(summary._load_partial(p)) == ["a"]


def test_sin_contenido_no_llama_al_modelo_para_el_resumen_general():
    llm = FakeLLM()
    empty = {k: [] for k in (*summary.SECTIONS, "tareas")}
    assert summary.overview(llm, empty) == "" and llm.calls == 0


@pytest.mark.slow
def test_resumen_real_con_qwen_4b(tmp_path):
    from transcriptor.config import get_settings

    if not summary.model_ready():
        pytest.skip("falta el modelo GGUF (transcriptor resumir lo descarga)")
    t = Transcript("es", 120, [
        Segment(0, 20, "Hola Gilda, hoy tenemos que decidir la fecha de entrega del informe.", speaker="Persona 1"),
        Segment(20, 40, "Yo creo que lo podemos entregar el viernes quince, ¿te parece bien?", speaker="Persona 2"),
        Segment(40, 60, "Perfecto, quedamos así. Yo me encargo de enviar el borrador el miércoles.", speaker="Persona 1"),
        Segment(60, 80, "Gilda, ¿quién revisa el presupuesto? Todavía no lo sabemos.", speaker="Persona 1"),
        Segment(80, 100, "Yo lo reviso. El presupuesto total es de cuatro mil dólares.", speaker="Persona 2"),
    ], speakers={"Persona 1": "Germán", "Persona 2": "Gilda"})
    out = summary.summarize(t, summary.LlamaLLM(get_settings()), tmp_path / "r.resumen.md")
    md = out.read_text(encoding="utf-8")
    assert "Resumen general" in md and "viernes" in md.lower()
