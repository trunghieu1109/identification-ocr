FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    OCR_SOURCE_DIR=/opt/vietnam-cccd-ocr \
    OCR_DEVICE=cpu \
    OCR_MODEL=vgg_transformer

RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgomp1 libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*

# Set OCR_REF to a commit SHA for a reproducible upstream checkout.
ARG OCR_REF=2d951ac603a43677572ff39ed6a21bdebe622263
RUN git clone https://github.com/namlhn/vietnam-cccd-ocr.git /opt/vietnam-cccd-ocr \
    && git -C /opt/vietnam-cccd-ocr checkout "$OCR_REF" \
    && git -C /opt/vietnam-cccd-ocr rev-parse HEAD > /opt/ocr-upstream-revision

WORKDIR /app
COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r /opt/vietnam-cccd-ocr/requirements.txt -r requirements.txt -r requirements-cpu.txt
COPY requirements-qr.txt ./
ENV QR_MODEL_DIR=/opt/qr-models
# OpenCV variants share cv2; install contrib last to keep WeChatQRCode available.
RUN pip install --no-cache-dir -r requirements-qr.txt \
    && pip install --no-cache-dir --force-reinstall --no-deps opencv-contrib-python==4.10.0.84
COPY patches ./patches
COPY scripts/patch-upstream.sh ./scripts/patch-upstream.sh
RUN sh scripts/patch-upstream.sh /opt/vietnam-cccd-ocr
COPY identification_demo ./identification_demo
COPY scripts/setup-qr-models.py ./scripts/setup-qr-models.py
RUN python scripts/setup-qr-models.py --directory /opt/qr-models
COPY scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh

ENV TMPDIR=/root/.cache/vietocr OCR_UPLOAD_TMP_DIR=/tmp
ENTRYPOINT ["sh", "/app/scripts/docker-entrypoint.sh"]

EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "identification_demo.web:app", "--host", "0.0.0.0", "--port", "8080"]
