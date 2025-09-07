# download_models.py
#!/usr/bin/env python3
"""
Smart model downloader for MuseTalk1.5
Downloads models on first run, caches for subsequent runs
"""

import os
import logging
import subprocess
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_CONFIGS = {
    # Only essential models for basic functionality
    "unet_model": {
        "url": "https://huggingface.co/kevinwang676/MuseTalk1.5/resolve/main/models/musetalk/pytorch_model.bin",
        "path": "/app/MuseTalk/models/musetalkV15/unet.pth",
        "size_mb": 2500,
        "required": True
    },
    "unet_config": {
        "url": "https://huggingface.co/kevinwang676/MuseTalk1.5/resolve/main/models/musetalk/musetalk.json",
        "path": "/app/MuseTalk/models/musetalkV15/musetalk.json", 
        "size_mb": 0.01,
        "required": True
    },
    "vae_model": {
        "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin",
        "path": "/app/MuseTalk/models/sd-vae/diffusion_pytorch_model.bin",
        "size_mb": 330,
        "required": True
    },
    "vae_config": {
        "url": "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json",
        "path": "/app/MuseTalk/models/sd-vae/config.json",
        "size_mb": 0.01,
        "required": True
    },
    "whisper_model": {
        "url": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
        "path": "/app/MuseTalk/models/whisper/tiny.pt",
        "size_mb": 39,
        "required": True
    },
    # Optional models - download on demand
    "dwpose_model": {
        "url": "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth",
        "path": "/app/MuseTalk/models/dwpose/dw-ll_ucoco_384.pth",
        "size_mb": 200,
        "required": False
    }
}

def check_disk_space(required_mb):
    """Check available disk space"""
    import shutil
    free_bytes = shutil.disk_usage('/app').free
    free_mb = free_bytes / (1024 * 1024)
    return free_mb > required_mb

def download_model_smart(model_name, config):
    """Smart model download with validation"""
    try:
        path = config["path"]
        
        # Check if already exists và valid
        if os.path.exists(path):
            file_size = os.path.getsize(path) / (1024 * 1024)
            expected_size = config["size_mb"]
            
            if abs(file_size - expected_size) < expected_size * 0.1:  # 10% tolerance
                logger.info(f"✅ {model_name} already exists and valid ({file_size:.1f}MB)")
                return True
        
        # Check disk space
        if not check_disk_space(config["size_mb"] * 1.5):  # 50% buffer
            logger.error(f"❌ Insufficient disk space for {model_name}")
            return False
        
        # Download
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        cmd = [
            "aria2c", "--console-log-level=error", 
            "-c", "-x", "16", "-s", "16", "-k", "1M",
            config["url"], "-d", os.path.dirname(path), 
            "-o", os.path.basename(path)
        ]
        
        logger.info(f"📥 Downloading {model_name} ({config['size_mb']:.1f}MB)...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info(f"✅ {model_name} downloaded successfully")
            return True
        else:
            logger.error(f"❌ {model_name} download failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error downloading {model_name}: {e}")
        return False

def download_models(required_only=False):
    """Download models based on mode"""
    logger.info("🔄 Starting model download...")
    
    total_downloaded = 0
    failed_downloads = []
    
    for model_name, config in MODEL_CONFIGS.items():
        if required_only and not config["required"]:
            logger.info(f"⏭️ Skipping optional model: {model_name}")
            continue
            
        success = download_model_smart(model_name, config)
        if success:
            total_downloaded += config["size_mb"]
        else:
            failed_downloads.append(model_name)
            if config["required"]:
                logger.error(f"❌ Required model {model_name} failed to download")
                return False
    
    logger.info(f"✅ Downloaded {total_downloaded:.1f}MB total")
    
    if failed_downloads:
        logger.warning(f"⚠️ Failed downloads: {failed_downloads}")
    
    # Create Whisper config if needed
    whisper_config_path = "/app/MuseTalk/models/whisper/config.json"
    if not os.path.exists(whisper_config_path):
        whisper_config = {
            "architectures": ["WhisperForConditionalGeneration"], 
            "model_type": "whisper"
        }
        with open(whisper_config_path, 'w') as f:
            import json
            json.dump(whisper_config, f)
    
    return len(failed_downloads) == 0 or all(not MODEL_CONFIGS[name]["required"] for name in failed_downloads)

if __name__ == "__main__":
    import sys
    required_only = "--required-only" in sys.argv
    success = download_models(required_only)
    sys.exit(0 if success else 1)
