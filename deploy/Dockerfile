FROM nvcr.io/nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv libreoffice-core libreoffice-writer libreoffice-calc \
        fonts-dejavu fonts-liberation libgl1 libglib2.0-0 tesseract-ocr tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /venv
ENV PATH=/venv/bin:$PATH
COPY requirements.txt /app/form2json/requirements.txt
RUN pip install -r /app/form2json/requirements.txt pytesseract \
 && pip install paddlepaddle-gpu==3.* -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
 && pip install "paddleocr>=3.0"
COPY . /app/form2json
WORKDIR /app
