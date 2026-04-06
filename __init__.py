"""
ComfyUI-FL-VoxCPM
A custom node pack for text-to-speech generation using VoxCPM models in ComfyUI

Features:
- VoxCPM 1.5 and 0.5B model support
- Voice cloning with reference audio
- LoRA fine-tuning support with auto rank detection
- Training nodes for custom voice adaptation

Based on VoxCPM architecture:
https://github.com/openbmb/VoxCPM

Author: FL
Version: 1.0.0
"""

import os
import sys
import logging
import folder_paths
from .modules.model_info import AVAILABLE_VOXCPM_MODELS, MODEL_CONFIGS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[ComfyUI-FL-VoxCPM] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

src_dir = os.path.join(current_dir, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

VOXCPM_SUBDIR_NAME = "VoxCPM"

tts_path = os.path.join(folder_paths.models_dir, "tts")
os.makedirs(tts_path, exist_ok=True)

if "tts" not in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["tts"] = ([tts_path], folder_paths.supported_pt_extensions)
else:
    if tts_path not in folder_paths.folder_names_and_paths["tts"][0]:
        folder_paths.folder_names_and_paths["tts"][0].append(tts_path)

for model_name, config in MODEL_CONFIGS.items():
    AVAILABLE_VOXCPM_MODELS[model_name] = {
        "type": "official",
        **config
    }

voxcpm_search_paths = []
for tts_folder in folder_paths.get_folder_paths("tts"):
    potential_path = os.path.join(tts_folder, VOXCPM_SUBDIR_NAME)
    if os.path.isdir(potential_path) and potential_path not in voxcpm_search_paths:
        voxcpm_search_paths.append(potential_path)

for search_path in voxcpm_search_paths:
    if not os.path.isdir(search_path):
        continue
    for item in os.listdir(search_path):
        item_path = os.path.join(search_path, item)
        if os.path.isdir(item_path) and item not in AVAILABLE_VOXCPM_MODELS:
            config_exists = os.path.exists(os.path.join(item_path, "config.json"))
            weights_exist = os.path.exists(os.path.join(item_path, "pytorch_model.bin")) or os.path.exists(os.path.join(item_path, "model.safetensors"))

            if config_exists and weights_exist:
                AVAILABLE_VOXCPM_MODELS[item] = {
                    "type": "local",
                    "path": item_path
                }

from .nodes import (
    FL_VoxCPM_TTS,
    FL_VoxCPM_V2_TTS,
    FL_VoxCPM_TrainConfig,
    FL_VoxCPM_V2_TrainConfig,
    FL_VoxCPM_DatasetMaker,
    FL_VoxCPM_LoRATrainer,
)

NODE_CLASS_MAPPINGS = {
    "FL_VoxCPM_TTS": FL_VoxCPM_TTS,
    "FL_VoxCPM_V2_TTS": FL_VoxCPM_V2_TTS,
    "FL_VoxCPM_TrainConfig": FL_VoxCPM_TrainConfig,
    "FL_VoxCPM_V2_TrainConfig": FL_VoxCPM_V2_TrainConfig,
    "FL_VoxCPM_DatasetMaker": FL_VoxCPM_DatasetMaker,
    "FL_VoxCPM_LoRATrainer": FL_VoxCPM_LoRATrainer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FL_VoxCPM_TTS": "FL VoxCPM TTS",
    "FL_VoxCPM_V2_TTS": "FL VoxCPM V2 TTS",
    "FL_VoxCPM_TrainConfig": "FL VoxCPM Train Config",
    "FL_VoxCPM_V2_TrainConfig": "FL VoxCPM V2 Train Config",
    "FL_VoxCPM_DatasetMaker": "FL VoxCPM Dataset Maker",
    "FL_VoxCPM_LoRATrainer": "FL VoxCPM LoRA Trainer",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print("\033[92m[ComfyUI-FL-VoxCPM] Loaded successfully!\033[0m")
print("\033[92m[ComfyUI-FL-VoxCPM] Nodes available: TTS, V2 TTS, Train Config, V2 Train Config, Dataset Maker, LoRA Trainer\033[0m")
