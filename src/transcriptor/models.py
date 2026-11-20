"""Modelos de datos del transcriptor."""

from dataclasses import dataclass, field, fields


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None  # id estable ("Persona 1"); el nombre real vive en Transcript.speakers
    words: list[dict] | None = None  # [{"start", "end", "word"}], solo con word_timestamps

    def to_dict(self) -> dict:
        """Como dict, omitiendo los campos opcionales vacíos (los JSON sin hablantes no cambian)."""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})  # tolera claves desconocidas


@dataclass
class Transcript:
    language: str
    duration: float
    segments: list[Segment] = field(default_factory=list)
    speakers: dict[str, str] = field(default_factory=dict)  # id -> nombre ("Persona 1" -> "Gilda")

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def label(self, speaker: str | None) -> str | None:
        """Nombre a mostrar para un id de hablante (el propio id si no tiene nombre)."""
        if speaker is None:
            return None
        return self.speakers.get(speaker) or speaker

    @property
    def has_speakers(self) -> bool:
        return any(s.speaker for s in self.segments)

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            language=d["language"],
            duration=d["duration"],
            segments=[Segment.from_dict(s) for s in d["segments"]],
            # Descarta nombres vacíos (o que solo traen un BOM invisible)
            speakers={k: v for k, v in (d.get("speakers") or {}).items() if v.replace("﻿", "").strip()},
        )
