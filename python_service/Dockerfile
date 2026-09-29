#python_service/Dockerfile

# CPU-only Dockerfile — no nvidia base image needed
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# STEP 1: Install CPU-only PyTorch explicitly from the PyTorch servers
RUN pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install your other packages from standard PyPI
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Model is mounted as a volume
RUN mkdir -p /app/models /tmp/defense_outputs

ENV MODEL_PATH=/app/models/defense_final.pth
ENV MODEL_VARIANT=aligned
ENV NUM_SECRETS=5
ENV IMAGE_SIZE=256
ENV TEXTURE_MIN=50.0
ENV OUTPUT_DIR=/tmp/defense_outputs

EXPOSE 5001

CMD ["python", "app.py"]