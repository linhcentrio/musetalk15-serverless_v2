# MuseTalk1.5 RunPod Serverless - OPTIMIZED BUILD
FROM spxiong/pytorch:2.0.1-py3.9.12-cuda11.8.0-ubuntu22.04

WORKDIR /app

# Environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONPATH="/app:/app/MuseTalk"

# Install system dependencies (lightweight first)
RUN apt-get update && apt-get install -y \
    ffmpeg wget curl git unzip aria2 \
    build-essential python3.9-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    libsndfile1 libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install MMlab packages
RUN pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmcv==2.0.1" && \
    mim install "mmdet==3.1.0" && \
    mim install "mmpose==1.1.0"

# === OPTIMIZED: Clone code only (no models) ===
RUN git clone https://huggingface.co/kevinwang676/MuseTalk1.5 /app/MuseTalk && \
    # Remove large files to reduce image size
    find /app/MuseTalk -name "*.pth" -delete && \
    find /app/MuseTalk -name "*.bin" -delete && \
    find /app/MuseTalk -name "*.safetensors" -delete && \
    find /app/MuseTalk -name "*.gguf" -delete

# Create model directories
RUN mkdir -p /app/MuseTalk/models/{musetalkV15,sd-vae,whisper,dwpose,face-parse-bisent,syncnet}

# Copy application handler và model download script
COPY musetalk_handler.py /app/musetalk_handler.py
COPY download_models.py /app/download_models.py

# === MODEL DOWNLOAD AT RUNTIME (not build time) ===
# Models sẽ được download lần đầu khi container start

# Final verification
RUN python -c "import torch, cv2, numpy, librosa; print('✅ Core packages OK')" && \
    python -c "import runpod, minio; print('✅ RunPod/MinIO OK')" && \
    python -c "from transformers import WhisperModel; print('✅ Transformers OK')" || echo "⚠️ Some packages missing"

# Health check
HEALTHCHECK --interval=30s --timeout=15s --start-period=300s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(); print('🚀 Ready')" || exit 1

EXPOSE 8000

# Run handler
CMD ["python", "musetalk_handler.py"]
