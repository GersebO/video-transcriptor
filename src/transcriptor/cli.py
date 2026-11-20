"""Interfaz de línea de comandos (Typer). Sin lógica de transcripción propia."""

import shutil
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import (
    BarColumn, Progress, TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)

from . import diarization, extractive, speakers, summary
from .audio import FFmpegNotFoundError, InsufficientDiskError, InvalidMediaError, extract_audio
from .config import get_settings
from .engine import TranscriptionEngine
from .formats import FORMATS, export, load_transcript
from .models import Transcript

app = typer.Typer(help="Transcriptor local de video y audio (todo corre en tu máquina).")
console = Console(stderr=True)

MODELS = ("tiny", "base", "small", "medium", "large-v3")
MEDIA_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".wmv", ".m4v", ".ts",
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
}

SUMMARY_MODES = ("ia", "extractivo")

EXIT_INPUT_ERROR = 1
EXIT_INTERNAL_ERROR = 2


def _expand_inputs(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in MEDIA_EXTENSIONS))
        else:
            files.append(p)
    return files


LAST_PATH_FILE = "ultima_ruta.txt"


def _clean_path(raw: str) -> Path:
    # Al arrastrar un archivo a la terminal de Windows llega entre comillas
    return Path(raw.lstrip("﻿").strip().strip("\"'").strip()).expanduser()


def _ask_path(models_dir: Path) -> Path:
    """Pregunta dónde está el video/audio/carpeta y repite hasta que exista."""
    last_file = models_dir / LAST_PATH_FILE
    try:
        last = last_file.read_text(encoding="utf-8").strip()
    except OSError:
        last = ""
    while True:
        question = "¿Dónde está el video, el audio o la carpeta? (pega la ruta)"
        raw = typer.prompt(question, default=last) if last else typer.prompt(question)
        path = _clean_path(raw)
        if path.exists():
            try:
                models_dir.mkdir(parents=True, exist_ok=True)
                last_file.write_text(str(path), encoding="utf-8")
            except OSError:
                pass
            return path
        console.print(f"[red]No existe: {path}[/red] Prueba de nuevo.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Sin subcomando, pregunta qué transcribir."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(transcribe, archivos=None, format=None, language=None, model=None,
                   output=None, device=None, stdout=False, hablantes=None, num_speakers=None, resumen=None)


@app.command()
def transcribe(
    archivos: Annotated[Optional[list[Path]], typer.Argument(help="Archivos de video/audio o carpetas. Si no das ninguno, te pregunta.")] = None,
    format: Annotated[list[str], typer.Option("--format", "-f", help="txt | srt | vtt | json (repetible).")] = None,
    language: Annotated[Optional[str], typer.Option("--language", "-l", help="es, en, ... (por defecto: detección automática).")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="tiny | base | small | medium | large-v3.")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Carpeta de salida (por defecto: junto al archivo).")] = None,
    device: Annotated[Optional[str], typer.Option("--device", help="auto | cpu | cuda.")] = None,
    stdout: Annotated[bool, typer.Option("--stdout", help="Imprime el resultado en vez de guardarlo.")] = False,
    hablantes: Annotated[Optional[bool], typer.Option("--hablantes/--sin-hablantes", help="Identifica quién habla (en modo interactivo se pregunta).")] = None,
    num_speakers: Annotated[Optional[int], typer.Option("--num-speakers", help="Cantidad de hablantes, si la sabes (mejora el resultado).")] = None,
    resumen: Annotated[Optional[str], typer.Option("--resumen", help="ia | extractivo | no. Genera <nombre>.resumen.md (en modo interactivo se pregunta).")] = None,
):
    """Transcribe uno o varios archivos (sin límite de tamaño ni duración)."""
    interactive = not archivos
    if interactive:
        archivos = [_ask_path(get_settings().models_dir)]
    formats = format or (["txt", "srt"] if interactive else ["txt"])
    if hablantes is None:
        hablantes = typer.confirm("¿Identificar quién habla?", default=True) if interactive else False
    for f in formats:
        if f not in FORMATS:
            console.print(f"[red]Formato no soportado: {f}. Usa: {', '.join(FORMATS)}[/red]")
            raise typer.Exit(EXIT_INPUT_ERROR)
    if resumen not in (None, *SUMMARY_MODES, "no"):
        console.print(f"[red]--resumen debe ser ia, extractivo o no (recibido: {resumen}).[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR)
    if resumen is None and interactive:
        resumen = ""
        while resumen not in (*SUMMARY_MODES, "no"):
            resumen = typer.prompt("¿Generar resumen? [ia/extractivo/no]", default="no").strip().lower()

    overrides = {k: v for k, v in {"model_size": model, "device": device}.items() if v}
    settings = get_settings().model_copy(update=overrides)
    files = _expand_inputs(archivos)
    if not files:
        console.print("[red]No se encontraron archivos para transcribir.[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR)

    engine = TranscriptionEngine.get(settings)
    exit_code = 0

    for path in files:
        try:
            _transcribe_one(engine, path, formats, language, output, stdout,
                            diarize=hablantes, num_speakers=num_speakers, ask_names=interactive,
                            summary_mode=resumen if resumen in SUMMARY_MODES else None)
        except (FFmpegNotFoundError, InvalidMediaError, InsufficientDiskError, ValueError) as e:
            console.print(f"[red]Error con {path.name}: {e}[/red]")
            exit_code = max(exit_code, EXIT_INPUT_ERROR)
        except Exception as e:  # noqa: BLE001 - error interno inesperado
            console.print(f"[red]Error interno con {path.name}: {e}[/red]")
            exit_code = EXIT_INTERNAL_ERROR
    raise typer.Exit(exit_code)


def _write_outputs(t: Transcript, out_dir: Path, stem: str, formats: list[str], stdout: bool = False) -> None:
    for fmt in formats:
        content = export(t, fmt)
        if stdout:
            sys.stdout.write(content if content.endswith("\n") else content + "\n")
        else:
            target = out_dir / f"{stem}.{fmt}"
            target.write_text(content, encoding="utf-8")
            console.print(f"  [green]✔[/green] {target}")


def _identify_speakers(t: Transcript, wav: Path, num_speakers: int | None) -> None:
    """Descarga los modelos de voz si faltan, separa las voces y las asigna a los segmentos."""
    settings = get_settings()
    if not diarization.models_ready(settings):
        console.print("Descargando los modelos de voz (una sola vez, ~45 MB)...")
        diarization.ensure_models(settings)
    with Progress(
        TextColumn("[cyan]Separando voces"), BarColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), console=console,
    ) as bar:
        task = bar.add_task("", total=1.0)
        turns = diarization.diarize(
            wav, num_speakers=num_speakers, settings=settings,
            on_progress=lambda done, total: bar.update(task, completed=done / total if total else 1.0),
        )
    diarization.assign_speakers(t, turns)
    console.print(f"  {len(speakers.speaker_ids(t))} hablante(s) detectado(s).")


def _ask_speaker_names(t: Transcript, media: Path | None) -> None:
    speakers.ask_names(t, media, ask=lambda q, d: typer.prompt(q, default=d), say=console.print)


def _transcribe_one(engine, path: Path, formats: list[str], language, output, stdout: bool,
                    diarize: bool = False, num_speakers: int | None = None, ask_names: bool = False,
                    summary_mode: str | None = None) -> None:
    out_dir = output or path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / f"{path.stem}.partial.jsonl"

    console.print(f"[bold]{path.name}[/bold]: extrayendo audio...")
    wav = extract_audio(path)
    try:
        with Progress(
            TextColumn("[cyan]Transcribiendo"), BarColumn(), TaskProgressColumn(),
            TimeElapsedColumn(), TimeRemainingColumn(), console=console,
        ) as bar:
            task = bar.add_task("", total=1.0)
            transcript = engine.transcribe(
                wav, language=language, word_timestamps=diarize,
                on_progress=lambda done, total: bar.update(task, completed=done / total if total else 1.0),
                partial_path=partial,
            )
        if diarize:  # el WAV se necesita aquí: se borra recién al terminar
            _identify_speakers(transcript, wav, num_speakers)
    finally:
        shutil.rmtree(wav.parent, ignore_errors=True)
    partial.unlink(missing_ok=True)

    if diarize:
        if ask_names:
            _ask_speaker_names(transcript, path)
        if "json" not in formats:  # los pasos siguientes (nombres, resumen) parten de este JSON
            formats = [*formats, "json"]
        if transcript.speakers:
            speakers.save_names(out_dir / f"{path.stem}.json", transcript.speakers)
    _write_outputs(transcript, out_dir, path.stem, formats, stdout)
    if summary_mode:
        _summarize(transcript, summary_mode, out_dir / f"{path.stem}.resumen.md", path.stem)


def _summarize(t: Transcript, mode: str, output: Path, title: str, model: str | None = None) -> None:
    if mode == "extractivo":
        extractive.summarize_extractive(t, output, title)
        console.print(f"  [green]✔[/green] {output}")
    else:
        _run_summary(t, output, title, model)


def _run_summary(t: Transcript, output: Path, title: str, model: str | None = None) -> None:
    """Descarga el modelo local si falta (con aviso de tamaño) y genera el resumen por tramos."""
    settings = get_settings()
    if model:
        settings = settings.model_copy(update={"llm_model": model})
    if not summary.model_ready(settings):
        console.print(f"Descargando el modelo de resumen {settings.llm_model} (una sola vez, varios GB)...")
        with Progress(
            TextColumn("[cyan]Descargando"), BarColumn(), TaskProgressColumn(), TimeRemainingColumn(), console=console,
        ) as bar:
            task = bar.add_task("", total=1.0)
            summary.ensure_model(
                settings, on_progress=lambda done, total: bar.update(task, completed=done / total if total else 0.0)
            )
    console.print("Cargando el modelo de resumen...")
    llm = summary.LlamaLLM(settings)
    with Progress(
        TextColumn("[cyan]Resumiendo"), BarColumn(), TaskProgressColumn(),
        TimeElapsedColumn(), TimeRemainingColumn(), console=console,
    ) as bar:
        task = bar.add_task("", total=1.0)
        summary.summarize(
            t, llm, output, title=title,
            on_progress=lambda done, total: bar.update(task, completed=done / total if total else 1.0),
        )
    console.print(f"  [green]✔[/green] {output}")


@app.command("resumir")
def resumir_cmd(
    transcript: Annotated[Path, typer.Argument(help="Transcripción ya hecha (.json o .srt).")],
    modo: Annotated[str, typer.Option("--modo", help="ia | extractivo.")] = "ia",
    modelo: Annotated[Optional[str], typer.Option("--modelo", help="Archivo .gguf dentro de la carpeta de modelos (por defecto el configurado).")] = None,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Archivo de salida (por defecto <nombre>.resumen.md).")] = None,
):
    """Resume una transcripción ya hecha, sin volver a transcribir."""
    if modo not in SUMMARY_MODES:
        console.print(f"[red]Modo no válido: {modo}. Usa ia o extractivo.[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR)
    if not transcript.exists():
        console.print(f"[red]No existe: {transcript}[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR)
    try:
        t = _load_with_names(transcript)
        _summarize(t, modo, output or transcript.with_name(f"{transcript.stem}.resumen.md"), transcript.stem, modelo)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR) from None
    except Exception as e:  # noqa: BLE001 - error interno inesperado (modelo, memoria...)
        console.print(f"[red]Error interno: {e}[/red]")
        raise typer.Exit(EXIT_INTERNAL_ERROR) from None


def _load_with_names(transcript_path: Path) -> Transcript:
    """Carga la transcripción y le aplica los nombres de su .speakers.json, si existe."""
    t = load_transcript(transcript_path)
    t.speakers = {**t.speakers, **speakers.load_names(transcript_path)}
    return t


def _save_transcript(t: Transcript, transcript_path: Path, formats: list[str]) -> None:
    """Guarda el JSON (junto con los nombres) y regenera los demás formatos a su lado."""
    out_dir, stem = transcript_path.parent, transcript_path.stem
    if t.speakers:
        speakers.save_names(transcript_path, t.speakers)
    _write_outputs(t, out_dir, stem, sorted({*formats, "json"}, key=FORMATS.index))


@app.command("hablantes")
def hablantes_cmd(
    video: Annotated[Path, typer.Argument(help="Video o audio de la reunión.")],
    transcript: Annotated[Path, typer.Option("--transcript", "-t", help="Transcripción ya hecha (.json o .srt).")],
    num_speakers: Annotated[Optional[int], typer.Option("--num-speakers", help="Cantidad de hablantes, si la sabes.")] = None,
    format: Annotated[list[str], typer.Option("--format", "-f", help="txt | srt | vtt (repetible).")] = None,
):
    """Identifica quién habla en una transcripción ya hecha, sin volver a transcribir."""
    formats = format or ["txt", "srt"]
    if not video.exists() or not transcript.exists():
        console.print("[red]No existe el video o la transcripción indicada.[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR)
    try:
        t = _load_with_names(transcript)
        console.print(f"[bold]{video.name}[/bold]: extrayendo audio...")
        wav = extract_audio(video)
        try:
            _identify_speakers(t, wav, num_speakers)
        finally:
            shutil.rmtree(wav.parent, ignore_errors=True)
        # Un .srt no guarda el idioma ni las palabras: el resultado se guarda como .json aparte
        target = transcript if transcript.suffix.lower() == ".json" else transcript.with_suffix(".json")
        _save_transcript(t, target, formats)  # primero se guarda: separar voces tarda y no se debe perder
        if sys.stdin.isatty():
            _ask_speaker_names(t, video)
            _save_transcript(t, target, formats)
        else:
            console.print("Sin terminal interactiva: ponles nombre después con `transcriptor nombres`.")
    except (FFmpegNotFoundError, InvalidMediaError, InsufficientDiskError, ValueError, FileNotFoundError) as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR) from None


@app.command("nombres")
def nombres_cmd(
    transcript: Annotated[Path, typer.Argument(help="Transcripción .json con hablantes.")],
    video: Annotated[Optional[Path], typer.Option("--video", help="Para poder escuchar las voces.")] = None,
    format: Annotated[list[str], typer.Option("--format", "-f", help="txt | srt | vtt (repetible).")] = None,
):
    """Cambia los nombres de los hablantes sin volver a separar voces."""
    if not transcript.exists():
        console.print(f"[red]No existe: {transcript}[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR)
    t = _load_with_names(transcript)
    if not t.has_speakers:
        console.print("[red]Esta transcripción no tiene hablantes. Ejecuta primero: transcriptor hablantes[/red]")
        raise typer.Exit(EXIT_INPUT_ERROR)
    _ask_speaker_names(t, video)
    _save_transcript(t, transcript, format or ["txt", "srt"])


def _is_downloaded(name: str, models_dir: Path) -> bool:
    return (models_dir / f"models--Systran--faster-whisper-{name}").exists()


@app.command()
def models(
    descargar_voz: Annotated[bool, typer.Option("--descargar-voz", help="Descarga los modelos de voz (~45 MB).")] = False,
):
    """Lista los modelos disponibles y cuáles están descargados."""
    settings = get_settings()
    if descargar_voz:
        diarization.ensure_models(settings)
    for name in MODELS:
        mark = "descargado" if _is_downloaded(name, settings.models_dir) else "no descargado"
        default = " (por defecto)" if name == settings.model_size else ""
        typer.echo(f"{name:<10} {mark}{default}")
    voz = "descargado" if diarization.models_ready(settings) else "no descargado"
    typer.echo(f"{'voz':<10} {voz} (separar hablantes, ~45 MB)")
    ia = "descargado" if summary.model_ready(settings) else "no descargado"
    typer.echo(f"{'resumen':<10} {ia} ({settings.llm_model})")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Dirección de escucha.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Puerto.")] = 8000,
):
    """Levanta la API HTTP."""
    import uvicorn

    uvicorn.run("transcriptor.api:app", host=host, port=port)


if __name__ == "__main__":
    app()
