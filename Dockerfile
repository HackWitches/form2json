# CPU (geometria + OCR); VLM (Ollama) jako osobny kontener, adres w FORM2JSON_VLM_URL
FROM python:3.12.8-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
        libreoffice-writer-nogui libreoffice-calc-nogui tesseract-ocr fonts-dejavu fonts-liberation \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 10001 app
WORKDIR /app
COPY requirements.txt /app/form2json/requirements.txt
RUN pip install --no-cache-dir -r /app/form2json/requirements.txt
COPY --chown=app:app . /app/form2json
USER app
ENTRYPOINT ["python", "-m", "form2json"]
