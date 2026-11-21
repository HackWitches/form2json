"""form2json - konwersja formularzy (PDF/DOC/DOCX/ODS/XLSX/JPG/PNG) do JSON."""
import os

# Limit pikseli dekodowanego obrazu (ochrona przed "bombą dekompresyjną": 50 KB PNG -> 2,5 GB RAM).
# OpenCV czyta tę zmienną tylko raz, więc musi być ustawiona przed pierwszym importem cv2.
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(80_000_000))
