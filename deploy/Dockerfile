# form2json - obraz aplikacji (CPU: LibreOffice, Tesseract, OpenCV). VLM działa w osobnym kontenerze
# Ollama na GPU (deploy/docker-compose.yml). Budowanie z katalogu form2json:
#   docker compose -f deploy/docker-compose.yml up --build
FROM python:3.11.9-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-pol \
        libreoffice-writer-nogui libreoffice-calc-nogui \
        fonts-dejavu fonts-liberation fonts-crosextra-carlito fonts-crosextra-caladea \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/form2json/requirements.txt
RUN pip install --no-cache-dir -r /app/form2json/requirements.txt
COPY . /app/form2json

# runtime.config z repozytorium ma ścieżki Windows - w kontenerze wejście/wyjście podaje docker-compose
ENTRYPOINT ["python", "-m", "form2json"]
CMD ["/data/in", "-o", "/data/out", "--ref", "/data/in"]
