FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Confirm OCR and OpenCV are available during Docker build
RUN tesseract --version
RUN python -c "import cv2; print('OpenCV:', cv2.__version__); print('CascadeClassifier:', hasattr(cv2, 'CascadeClassifier'))"
RUN python -c "import pytesseract; print('Tesseract path:', pytesseract.pytesseract.tesseract_cmd); print(pytesseract.get_tesseract_version())"

RUN python manage.py collectstatic --noinput || true

EXPOSE 10000

CMD ["sh", "-c", "python manage.py migrate && gunicorn borderguard.wsgi:application --bind 0.0.0.0:10000 --timeout 120"]