import json

import pytest

from transcriptor import formats
from transcriptor.models import Segment, Transcript


@pytest.fixture
def t():
    return Transcript("es", 7.5, [Segment(0.0, 2.5, " Hola a todos"), Segment(3661.007, 3663.2, "Segunda línea")])


def test_srt(t):
    assert formats.to_srt(t) == (
        "1\n00:00:00,000 --> 00:00:02,500\nHola a todos\n\n"
        "2\n01:01:01,007 --> 01:01:03,200\nSegunda línea\n"
    )


def test_vtt(t):
    assert formats.to_vtt(t) == (
        "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nHola a todos\n\n"
        "01:01:01.007 --> 01:01:03.200\nSegunda línea\n"
    )


def test_txt(t):
    assert formats.to_txt(t) == "Hola a todos\nSegunda línea\n"


def test_json(t):
    d = json.loads(formats.to_json(t))
    assert d["language"] == "es" and len(d["segments"]) == 2
    assert d["text"] == "Hola a todos Segunda línea"


def test_horas_largas():
    assert formats._timestamp(100 * 3600 + 1.5, ",") == "100:00:01,500"


def test_formato_invalido(t):
    with pytest.raises(ValueError):
        formats.export(t, "pdf")
