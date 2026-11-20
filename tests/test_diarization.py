import json
import subprocess
import sys
import wave

import numpy as np
import pytest

from transcriptor import cli, diarization, formats, speakers
from transcriptor.diarization import Turn, absorb_minor_speakers, assign_speakers, clean_turns
from transcriptor.models import Segment, Transcript


def words(*items):
    return [{"start": a, "end": b, "word": w} for a, b, w in items]


def test_asigna_el_hablante_con_mayor_solape():
    t = Transcript("es", 10, [Segment(0, 4, "uno"), Segment(5, 9, "dos"), Segment(3.5, 6, "mitad")])
    turns = [Turn(0, 4.5, "Persona 1"), Turn(4.5, 10, "Persona 2")]
    assign_speakers(t, turns)
    assert [s.speaker for s in t.segments] == ["Persona 1", "Persona 2", "Persona 2"]


def test_segmento_sin_solape_toma_el_turno_mas_cercano():
    t = Transcript("es", 10, [Segment(5.0, 5.5, "hueco")])
    assign_speakers(t, [Turn(0, 4, "Persona 1"), Turn(7, 9, "Persona 2")])
    assert t.segments[0].speaker == "Persona 1"


def test_divide_el_segmento_por_palabras_si_cruza_dos_hablantes():
    seg = Segment(
        0, 4, "hola que tal bien gracias",
        words=words((0, 0.5, " hola"), (0.5, 1, " que"), (1, 2, " tal"), (2, 3, " bien"), (3, 4, " gracias")),
    )
    t = Transcript("es", 4, [seg])
    assign_speakers(t, [Turn(0, 2, "Persona 1"), Turn(2, 4, "Persona 2")])
    assert [(s.speaker, s.text) for s in t.segments] == [
        ("Persona 1", "hola que tal"),
        ("Persona 2", "bien gracias"),
    ]
    assert t.segments[1].start == 2 and t.segments[1].end == 4


def test_sin_palabras_no_divide_y_elige_el_dominante():
    t = Transcript("es", 4, [Segment(0, 4, "frase larga")])
    assign_speakers(t, [Turn(0, 2.2, "Persona 1"), Turn(2.2, 4, "Persona 2")])
    assert len(t.segments) == 1 and t.segments[0].speaker == "Persona 1"


def test_turnos_cortos_se_funden_con_el_vecino():
    turns = [
        Turn(0, 5, "Persona 1"),
        Turn(5, 5.3, "Persona 2"),  # salto falso de 0,3 s
        Turn(5.3, 9, "Persona 1"),
        Turn(9, 12, "Persona 2"),
    ]
    assert clean_turns(turns) == [Turn(0, 9, "Persona 1"), Turn(9, 12, "Persona 2")]


def test_primer_turno_corto_pasa_al_siguiente():
    assert clean_turns([Turn(0, 0.2, "Persona 2"), Turn(0.2, 5, "Persona 1")]) == [Turn(0, 5, "Persona 1")]


def test_hablantes_con_muy_poco_habla_se_absorben_en_el_cercano():
    turns = [
        Turn(0, 60, "Persona 1"),
        Turn(60, 62, "Persona 2"),  # ruido de agrupado: 2 s
        Turn(62, 100, "Persona 3"),
        Turn(100, 103, "Persona 4"),  # ruido: 3 s, más cerca de Persona 3
        Turn(103, 130, "Persona 1"),
    ]
    got = absorb_minor_speakers(turns)
    # Persona 3 (38 s) y Persona 1 son principales; renumera por orden de aparición
    assert [(t.start, t.end, t.speaker_id) for t in got] == [
        (0, 62, "Persona 1"),
        (62, 103, "Persona 2"),
        (103, 130, "Persona 1"),
    ]


def test_si_todos_son_principales_no_cambia_nada():
    turns = [Turn(0, 30, "Persona 1"), Turn(30, 60, "Persona 2")]
    assert absorb_minor_speakers(turns) == turns


# --- formatos con hablantes ---

@pytest.fixture
def hablado():
    t = Transcript(
        "es", 10,
        [
            Segment(0, 2, "Hola", speaker="Persona 1"),
            Segment(2, 4, "¿Cómo estás?", speaker="Persona 1"),
            Segment(4, 6, "Bien", speaker="Persona 2"),
        ],
        speakers={"Persona 1": "Gilda"},
    )
    return t


def test_txt_agrupa_lineas_del_mismo_hablante(hablado):
    assert formats.to_txt(hablado) == "Gilda: Hola ¿Cómo estás?\nPersona 2: Bien\n"


def test_srt_y_vtt_llevan_el_nombre(hablado):
    assert "[Gilda] Hola" in formats.to_srt(hablado) and "[Persona 2] Bien" in formats.to_srt(hablado)
    assert "<v Gilda>Hola" in formats.to_vtt(hablado)


def test_sin_hablantes_la_salida_no_cambia():
    t = Transcript("es", 5, [Segment(0, 2, "Hola"), Segment(2, 4, "mundo")])
    assert formats.to_txt(t) == "Hola\nmundo\n"
    assert "speaker" not in formats.to_json(t) and "words" not in formats.to_json(t)
    assert "[" not in formats.to_srt(t)


def test_json_con_hablantes_ida_y_vuelta(hablado):
    back = Transcript.from_dict(json.loads(formats.to_json(hablado)))
    assert back.segments[2].speaker == "Persona 2" and back.speakers == {"Persona 1": "Gilda"}


def test_json_antiguo_y_claves_desconocidas_cargan():
    t = Transcript.from_dict(
        {"language": "es", "duration": 2, "segments": [{"start": 0, "end": 1, "text": "x", "futuro": 1}]}
    )
    assert t.segments[0].speaker is None and not t.has_speakers


def test_parse_srt():
    srt = "1\n00:00:00,000 --> 00:00:02,500\nHola\n\n2\n01:01:01,007 --> 01:01:03,200\nSegunda\nlínea\n"
    t = formats.parse_srt(srt)
    assert [(s.start, s.end, s.text) for s in t.segments] == [(0.0, 2.5, "Hola"), (3661.007, 3663.2, "Segunda línea")]
    assert t.duration == 3663.2


# --- nombres ---

def test_ejemplos_repartidos_y_nombres(tmp_path, hablado):
    hablado.segments.append(Segment(7, 9, "Esta es una frase larga de Persona 2", speaker="Persona 2"))
    ex = speakers.examples(hablado, "Persona 2")
    assert ex[0][1] == "Esta es una frase larga de Persona 2"  # prefiere frases largas
    assert speakers.speaker_ids(hablado) == ["Persona 1", "Persona 2"]
    path = tmp_path / "reunion.json"
    speakers.save_names(path, {"Persona 1": "Gilda"})
    assert speakers.names_path(path).name == "reunion.speakers.json"
    assert speakers.load_names(path) == {"Persona 1": "Gilda"}


def test_nombre_vacio_o_solo_bom_se_ignora(tmp_path):
    path = tmp_path / "r.json"
    speakers.names_path(path).write_text('{"Persona 1": "\\ufeff", "Persona 2": "Ana"}', encoding="utf-8")
    assert speakers.load_names(path) == {"Persona 2": "Ana"}


def test_comando_nombres_regenera_txt_y_srt(tmp_path, hablado):
    from typer.testing import CliRunner

    hablado.speakers = {}
    path = tmp_path / "reunion.json"
    path.write_text(formats.to_json(hablado), encoding="utf-8")
    r = CliRunner().invoke(cli.app, ["nombres", str(path)], input="Germán\nGilda\n")
    assert r.exit_code == 0, r.output
    assert (tmp_path / "reunion.txt").read_text(encoding="utf-8") == "Germán: Hola ¿Cómo estás?\nGilda: Bien\n"
    assert "[Gilda] Bien" in (tmp_path / "reunion.srt").read_text(encoding="utf-8")
    assert speakers.load_names(path) == {"Persona 1": "Germán", "Persona 2": "Gilda"}


def test_nombres_sin_hablantes_sale_con_1(tmp_path):
    from typer.testing import CliRunner

    path = tmp_path / "r.json"
    path.write_text(formats.to_json(Transcript("es", 1, [Segment(0, 1, "x")])), encoding="utf-8")
    assert CliRunner().invoke(cli.app, ["nombres", str(path)]).exit_code == 1


# --- diarización real con voces SAPI de Windows ---

@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "win32", reason="usa las voces SAPI de Windows")
def test_diarizacion_real_con_dos_voces_sapi(tmp_path):
    if not diarization.models_ready():
        pytest.skip("faltan los modelos de voz (transcriptor models --descargar-voz)")
    script = tmp_path / "voces.ps1"
    script.write_text(
        "param($out)\nAdd-Type -AssemblyName System.Speech\n$s = New-Object System.Speech.Synthesis.SpeechSynthesizer\n"
        "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,"
        "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,[System.Speech.AudioFormat.AudioChannel]::Mono)\n"
        "$a=@(@('Microsoft Helena Desktop','Buenos días, vamos a revisar el avance del proyecto de esta semana.'),"
        "@('Microsoft David Desktop','Sure, I finished the first module and the tests are passing now.'),"
        "@('Microsoft Helena Desktop','Perfecto, entonces queda pendiente la documentación para el viernes.'),"
        "@('Microsoft David Desktop','I will take care of the documentation and send it before the deadline.'),"
        "@('Microsoft Helena Desktop','Muy bien, gracias. Hablamos de nuevo la próxima semana.'),"
        "@('Microsoft David Desktop','Great, see you next week and have a nice weekend.'))\n"
        "$i=0; foreach($t in $a){ $s.SelectVoice($t[0]); $s.SetOutputToWaveFile(\"$out\\t$i.wav\",$f); "
        "$s.Speak($t[1]); $s.SetOutputToNull(); $i++ }\n",
        encoding="utf-8-sig",
    )
    r = subprocess.run(
        ["powershell", "-NoProfile", "-File", str(script), str(tmp_path)], capture_output=True, text=True
    )
    if r.returncode != 0 or not (tmp_path / "t5.wav").exists():
        pytest.skip(f"no hay voces SAPI utilizables: {r.stderr[:200]}")

    gap = np.zeros(8000, dtype=np.int16)
    parts, segs, pos = [], [], 0.0
    for i in range(6):
        with wave.open(str(tmp_path / f"t{i}.wav")) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        segs.append(Segment(pos, pos + len(x) / 16000, f"frase {i}"))
        pos += len(x) / 16000 + 0.5
        parts += [x, gap]
    mix = tmp_path / "mix.wav"
    with wave.open(str(mix), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(np.concatenate(parts).tobytes())

    t = Transcript("es", pos, segs)
    assign_speakers(t, diarization.diarize(mix))
    got = [s.speaker for s in t.segments]
    assert len(set(got)) == 2
    expected = [got[0], got[1]] * 3  # voces alternadas
    assert sum(a == b for a, b in zip(got, expected)) / 6 > 0.9
