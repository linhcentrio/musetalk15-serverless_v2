# musetalk_handler_optimized.py
#!/usr/bin/env python3
"""
MuseTalk Optimized Handler cho RunPod Serverless
Lazy model loading + smart caching
"""

import runpod
import os
import tempfile
import uuid
import requests
import time
import torch
import logging
import sys
import threading
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add MuseTalk paths
sys.path.insert(0, '/app/MuseTalk')

# Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_cache = {}
model_lock = threading.Lock()
models_downloaded = False

def ensure_models_downloaded():
    """Ensure models are downloaded (lazy download)"""
    global models_downloaded
    
    if models_downloaded:
        return True
    
    with model_lock:
        if models_downloaded:  # Double-check locking
            return True
            
        logger.info("📥 Downloading models on first request...")
        
        try:
            # Download essential models only
            result = subprocess.run([
                "python", "/app/download_models.py", "--required-only"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Models downloaded successfully")
                models_downloaded = True
                return True
            else:
                logger.error(f"❌ Model download failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Model download timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Model download error: {e}")
            return False

def load_models_lazy():
    """Lazy load models only when needed"""
    if 'models_loaded' in model_cache:
        logger.info("✅ Using cached models")
        return
    
    with model_lock:
        if 'models_loaded' in model_cache:
            return
            
        try:
            # Ensure models are downloaded first
            if not ensure_models_downloaded():
                raise RuntimeError("Failed to download required models")
            
            logger.info("🔄 Loading MuseTalk models...")
            
            # Import here to avoid import errors if models not downloaded
            from musetalk.utils.utils import load_all_model
            
            # Load models
            audio_processor, vae, unet, pe = load_all_model()
            
            # Half precision optimization
            pe = pe.half()
            vae.vae = vae.vae.half()  
            unet.model = unet.model.half()
            
            # Cache models
            model_cache.update({
                'models_loaded': True,
                'audio_processor': audio_processor,
                'vae': vae,
                'unet': unet,
                'pe': pe
            })
            
            logger.info("✅ Models loaded and cached")
            
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            raise e

# Import other components lazily
def get_musetalk_components():
    """Lazy import MuseTalk components"""
    try:
        from musetalk.utils.utils import get_file_type, get_video_fps, datagen
        from musetalk.utils.preprocessing import get_landmark_and_bbox, read_imgs, coord_placeholder
        from musetalk.utils.blending import get_image, get_image_prepare_material, get_image_blending
        return True
    except ImportError as e:
        logger.error(f"❌ MuseTalk components not available: {e}")
        return False

@torch.no_grad()
def handler(job):
    """
    Optimized handler với lazy loading
    """
    job_id = job.get("id", "unknown")
    start_time = time.time()
    
    try:
        # Lazy load models on first request
        load_models_lazy()
        
        # Import components
        if not get_musetalk_components():
            return {"error": "MuseTalk components not available", "status": "failed"}
        
        # ... rest of handler logic (same as before)
        # Using cached models from model_cache
        
        return {
            "output_video_url": "...",
            "processing_time_seconds": round(time.time() - start_time, 2),
            "status": "completed",
            "optimization_info": {
                "lazy_loading": True,
                "models_cached": 'models_loaded' in model_cache,
                "download_time_saved": "Models pre-downloaded" if models_downloaded else "Downloaded on demand"
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Handler error: {e}")
        return {
            "error": str(e),
            "status": "failed",
            "job_id": job_id
        }

def health_check():
    """Health check without forcing model download"""
    try:
        # Basic checks only
        if not torch.cuda.is_available():
            return False, "CUDA not available"
        
        # Don't download models during health check
        return True, "System ready (models will download on first request)"
        
    except Exception as e:
        return False, f"Health check failed: {str(e)}"

if __name__ == "__main__":
    logger.info("🚀 Starting MuseTalk Optimized Handler...")
    logger.info("💡 Models will download on first request (lazy loading)")
    
    try:
        # Start RunPod worker
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        sys.exit(1)
