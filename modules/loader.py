import os
import torch
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

import folder_paths

from ..src.voxcpm.core import VoxCPM
from ..src.voxcpm.model.voxcpm import LoRAConfig
from .model_info import AVAILABLE_VOXCPM_MODELS

logger = logging.getLogger(__name__)

LOADED_MODELS_CACHE = {}


def detect_lora_rank(lora_path: str) -> int:
    """
    Detect the LoRA rank from a safetensors or checkpoint file by inspecting tensor shapes.
    Returns the detected rank, or 32 as default if detection fails.
    """
    lora_path = Path(lora_path)

    try:
        if lora_path.suffix == ".safetensors":
            from safetensors import safe_open
            with safe_open(str(lora_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    if "lora_A" in key or "lora_a" in key:
                        tensor = f.get_tensor(key)
                        # lora_A has shape (r, in_features), so dimension 0 is the rank
                        return tensor.shape[0]
                    elif "lora_B" in key or "lora_b" in key:
                        tensor = f.get_tensor(key)
                        # lora_B has shape (out_features, r), so dimension 1 is the rank
                        return tensor.shape[1]
        elif lora_path.suffix in [".ckpt", ".pth", ".pt"]:
            state_dict = torch.load(str(lora_path), map_location="cpu", weights_only=True)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            for key, tensor in state_dict.items():
                if "lora_A" in key or "lora_a" in key:
                    return tensor.shape[0]
                elif "lora_B" in key or "lora_b" in key:
                    return tensor.shape[1]
    except Exception as e:
        logger.warning(f"Failed to detect LoRA rank from {lora_path}: {e}")

    logger.info("Could not detect LoRA rank, using default: 32")
    return 32


class VoxCPMModelHandler(torch.nn.Module):
    """
    A lightweight handler for a VoxCPM model. It acts as a container
    that ComfyUI's ModelPatcher can manage, while the actual heavy model
    is loaded on demand.
    """
    def __init__(self, model_name: str, lora_rank: int = 32):
        super().__init__()
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.model = None  # This will hold the actual loaded VoxCPM instance
        # Estimate size (VoxCPM1.5 is ~800M params in bf16 -> ~1.6GB + buffers)
        # We allocate 2.5GB to be safe for offloading calculations
        self.size = int(2.5 * (1024**3))

class VoxCPMLoader:
    @staticmethod
    def load_model(model_name: str, lora_rank: int = 32):
        """
        Loads a VoxCPM model, downloading it if necessary. Caches the loaded model instance.
        """
        cache_key = f"{model_name}_r{lora_rank}"
        if cache_key in LOADED_MODELS_CACHE:
            logger.info(f"Using cached VoxCPM model instance: {model_name} (rank={lora_rank})")
            return LOADED_MODELS_CACHE[cache_key]

        model_info = AVAILABLE_VOXCPM_MODELS.get(model_name)
        if not model_info:
            raise ValueError(f"Model '{model_name}' not found. Available models: {list(AVAILABLE_VOXCPM_MODELS.keys())}")

        voxcpm_path = None

        if model_info["type"] == "local":
            voxcpm_path = model_info["path"]
            logger.info(f"Loading local model from: {voxcpm_path}")

        elif model_info["type"] == "official":
            base_tts_path = os.path.join(folder_paths.get_folder_paths("tts")[0])
            voxcpm_models_dir = os.path.join(base_tts_path, "VoxCPM")
            os.makedirs(voxcpm_models_dir, exist_ok=True)
            
            voxcpm_path = os.path.join(voxcpm_models_dir, model_name)
            
            has_bin = os.path.exists(os.path.join(voxcpm_path, "pytorch_model.bin"))
            has_safe = os.path.exists(os.path.join(voxcpm_path, "model.safetensors"))
            
            if not (has_bin or has_safe):
                logger.info(f"Downloading official VoxCPM model '{model_name}' from {model_info['repo_id']}...")
                snapshot_download(
                    repo_id=model_info["repo_id"],
                    local_dir=voxcpm_path,
                    local_dir_use_symlinks=False,
                )

        if not voxcpm_path:
             raise RuntimeError(f"Could not determine path for model '{model_name}'")

        logger.info(f"Instantiating VoxCPM model with LoRA rank={lora_rank}...")

        lora_config = LoRAConfig(
            enable_lm=True,
            enable_dit=True,
            enable_proj=False,
            r=lora_rank,
            alpha=lora_rank // 2 if lora_rank > 1 else 1
        )

        model_instance = VoxCPM(
            voxcpm_model_path=voxcpm_path,
            enable_denoiser=False,
            optimize=False,
            lora_config=lora_config
        )

        LOADED_MODELS_CACHE[cache_key] = model_instance
        return model_instance