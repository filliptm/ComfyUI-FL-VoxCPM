import os
import logging
import folder_paths
from comfy_api.latest import io

from ..modules.model_info import AVAILABLE_VOXCPM_MODELS

logger = logging.getLogger(__name__)

VoxCPMTrainConfig = io.Custom("VOXCPM_TRAIN_CONFIG")
VoxCPMDataset = io.Custom("VOXCPM_DATASET")

# The training module imports 'argbind' and 'datasets'.
# We wrap this so the main inference node works without them.
TRAINING_IMPORT_ERROR = None
try:
    from ..modules.trainer import run_lora_training
except ImportError as e:
    run_lora_training = None
    TRAINING_IMPORT_ERROR = str(e)
    # Check specifically for the likely missing packages to give a better error
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
            description="Trains a LoRA adapter for VoxCPM.",
            is_output_node=True,
            inputs=[
                io.Combo.Input("base_model_name", options=model_names, default=model_names[0], tooltip="Base VoxCPM model to fine-tune."),
                VoxCPMTrainConfig.Input("train_config", tooltip="Configuration dictionary from VoxCPM Train Config node."),
                VoxCPMDataset.Input("dataset_path", tooltip="Path to the train.jsonl file from VoxCPM Dataset Maker."),
                io.String.Input("output_name", default="my_lora_v1", tooltip="Name of the subfolder in 'models/loras' to save results."),
                io.Int.Input("max_steps", default=1000, min=100, max=100000, tooltip="Total number of training steps."),
                io.Int.Input("save_every_steps", default=200, min=50, max=5000, tooltip="Save checkpoint every N steps."),
                io.Int.Input("num_workers", default=0, min=0, max=8, tooltip="Number of dataloader workers (0 for main thread)."),
            ],
            outputs=[
                io.String.Output(display_name="LoRA Output Path"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, base_model_name, train_config, dataset_path, output_name, max_steps, save_every_steps, num_workers):
        # Guard: Check if training module loaded successfully
        if run_lora_training is None:
            raise RuntimeError(f"Training functionality unavailable. {TRAINING_IMPORT_ERROR}")

        node_id = cls.hidden.unique_id

        # Determine output directory using ComfyUI's standard paths
        lora_base_dir = folder_paths.get_folder_paths("loras")[0]
        output_dir = os.path.join(lora_base_dir, output_name)

        try:
            # Delegate to trainer module
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
            )
            return io.NodeOutput(final_output_dir)
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise e
