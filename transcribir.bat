@echo off
rem Doble clic: activa el entorno y pregunta qué video transcribir.
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"
python -m transcriptor
echo.
pause
