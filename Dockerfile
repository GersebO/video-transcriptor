FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Pesos del modelo y base de datos/subidas en volúmenes para no repetir descargas
# ni perder trabajos al reiniciar el contenedor.
ENV TRANSCRIPTOR_MODELS_DIR=/models \
    TRANSCRIPTOR_JOBS_DB=/data/jobs.db
VOLUME ["/models", "/data"]

EXPOSE 8000
CMD ["transcriptor", "serve", "--host", "0.0.0.0", "--port", "8000"]
