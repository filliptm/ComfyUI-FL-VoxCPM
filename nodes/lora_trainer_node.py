import os
import logging
import folder_paths
from comfy_api.latest import io

from ..modules.model_info import AVAILABLE_VOXCPM_MODELS, V2_MODEL_NAMES

logger = logging.getLogger(__name__)

VoxCPMDataset = io.Custom("VOXCPM_DATASET")

# The training module imports 'argbind' and 'datasets'.
# We wrap this so the main inference node works without them.
TRAINING_IMPORT_ERROR = None
run_lora_training = None
run_lora_training_v2 = None
try:
    from ..modules.trainer import run_lora_training
except ImportError as e:
    TRAINING_IMPORT_ERROR = str(e)
    missing = []
    try:
        import argbind
    except ImportError:
        missing.append("argbind")
    try:
        import datasets
    except ImportError:
        missing.append("datasets")
    if missing:
        TRAINING_IMPORT_ERROR = f"Missing required packages for training: {', '.join(missing)}. Please run: pip install {' '.join(missing)}"

try:
    from ..modules.trainer_v2 import run_lora_training_v2
except ImportError:
    pass  # V2 trainer may have additional deps


class FL_VoxCPM_LoRATrainer(io.ComfyNode):
    CATEGORY = "FL/VoxCPM/Training"

    @classmethod
    def define_schema(cls) -> io.Schema:
        model_names = list(AVAILABLE_VOXCPM_MODELS.keys())
        if not model_names:
            model_names.append("No models found.")

        return io.Schema(
            node_id="FL_VoxCPM_LoRATrainer",
            display_name="FL VoxCPM LoRA Trainer",
            category=cls.CATEGORY,
            description="Trains a LoRA adapter for VoxCPM (V1 or V2). Auto-detects model version and routes to the appropriate trainer.",
            is_output_node=True,
            inputs=[
                io.Combo.Input("base_model_name", options=model_names, default=model_names[0], tooltip="Base VoxCPM model to fine-tune (V1 or V2)."),
                io.AnyType.Input("train_config", tooltip="Configuration from VoxCPM Train Config (V1) or V2 Train Config node."),
                VoxCPMDataset.Input("dataset_path", tooltip="Path to the train.jsonl file from VoxCPM Dataset Maker."),
                io.String.Input("output_name", default="my_lora_v1", tooltip="Name of the subfolder in 'models/loras/VoxCPM/' to save results."),
                io.Int.Input("max_steps", default=1000, min=100, max=100000, tooltip="Total number of training steps."),
                io.Int.Input("save_every_steps", default=200, min=50, max=5000, tooltip="Save checkpoint every N steps."),
                io.Int.Input("num_workers", default=0, min=0, max=8, tooltip="Number of dataloader workers (0 for main thread)."),
                io.String.Input("validation_text", multiline=True, optional=True, default="", tooltip="Text to synthesize at each checkpoint for audio validation. Leave empty to skip."),
                io.Int.Input("validation_steps", default=10, min=1, max=100, optional=True, tooltip="Inference timesteps for validation audio generation."),
            ],
            outputs=[
                io.String.Output(display_name="LoRA Output Path"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, base_model_name, train_config, dataset_path, output_name, max_steps, save_every_steps, num_workers,
                validation_text="", validation_steps=10):
        node_id = cls.hidden.unique_id
        is_v2_model = base_model_name in V2_MODEL_NAMES

        # Validate config version matches model version
        config_version = train_config.get("_version", "v1") if isinstance(train_config, dict) else "v1"
        if is_v2_model and config_version != "v2":
            raise ValueError(f"V2 model '{base_model_name}' requires a V2 Train Config node. Please connect FL VoxCPM V2 Train Config.")
        if not is_v2_model and config_version == "v2":
            raise ValueError(f"V1 model '{base_model_name}' requires a V1 Train Config node. Please connect FL VoxCPM Train Config.")

        # LoRA output under VoxCPM subdirectory
        lora_base_dir = folder_paths.get_folder_paths("loras")[0]
        output_dir = os.path.join(lora_base_dir, "VoxCPM", output_name)

        # Clean validation text
        val_text = validation_text.strip() if validation_text else None

        if is_v2_model:
            if run_lora_training_v2 is None:
                raise RuntimeError("V2 training functionality unavailable. Check that all V2 dependencies are installed.")
            try:
                final_output_dir = run_lora_training_v2(
                    base_model_name=base_model_name,
                    train_config=train_config,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    max_steps=max_steps,
                    save_every_steps=save_every_steps,
                    num_workers=num_workers,
                    output_name=output_name,
                    folder_paths_module=folder_paths,
                    node_id=node_id,
                    validation_text=val_text,
                    validation_steps=validation_steps,
                )
                return io.NodeOutput(final_output_dir)
            except Exception as e:
                logger.error(f"V2 Training failed: {e}")
                raise e
        else:
            if run_lora_training is None:
                raise RuntimeError(f"Training functionality unavailable. {TRAINING_IMPORT_ERROR}")
            try:
                final_output_dir = run_lora_training(
                    base_model_name=base_model_name,
                    train_config=train_config,
                    dataset_path=dataset_path,
                    output_dir=output_dir,
                    max_steps=max_steps,
                    save_every_steps=save_every_steps,
                    num_workers=num_workers,
                    output_name=output_name,
                    folder_paths_module=folder_paths,
                    node_id=node_id,
                    validation_text=val_text,
                    validation_steps=validation_steps,
                )
                return io.NodeOutput(final_output_dir)
            except Exception as e:
                logger.error(f"Training failed: {e}")
                raise e
