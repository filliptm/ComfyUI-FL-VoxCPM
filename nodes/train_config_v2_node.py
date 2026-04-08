import logging
from comfy_api.latest import io

logger = logging.getLogger(__name__)

VoxCPMV2TrainConfig = io.Custom("VOXCPM_V2_TRAIN_CONFIG")


class FL_VoxCPM_V2_TrainConfig(io.ComfyNode):
    CATEGORY = "FL/VoxCPM/Training"

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="FL_VoxCPM_V2_TrainConfig",
            display_name="FL VoxCPM V2 Train Config",
            category=cls.CATEGORY,
            description="Configuration parameters for VoxCPM V2 LoRA training (48kHz, 2B model). Defaults match official OpenBMB config.",
            inputs=[
                io.Float.Input("learning_rate", default=1e-4, min=1e-6, max=1e-2, step=1e-5, tooltip="Learning rate. Official default: 1e-4 for LoRA."),
                io.Int.Input("lora_rank", default=32, min=4, max=128, step=4, tooltip="LoRA rank. Official default: 32."),
                io.Int.Input("lora_alpha", default=32, min=1, max=128, step=1, tooltip="LoRA alpha. Official default: 32 (same as rank, scaling=1.0)."),
                io.Float.Input("lora_dropout", default=0.0, min=0.0, max=0.5, step=0.05, tooltip="Dropout probability for LoRA layers."),
                io.Int.Input("warmup_steps", default=100, min=0, max=1000, tooltip="Number of warmup steps for learning rate scheduler."),
                io.Int.Input("grad_accum_steps", default=8, min=1, max=64, tooltip="Gradient accumulation steps. Official default: 8 (effective batch=16 with batch_size=2)."),
                io.Int.Input("max_batch_tokens", default=8192, min=1024, max=32768, tooltip="Maximum number of tokens per batch."),
                io.Int.Input("sample_rate", default=16000, min=16000, max=48000, tooltip="Sample rate for audio encoding. V2 AudioVAE encodes at 16kHz (decodes to 48kHz via super-resolution)."),
                io.Float.Input("weight_decay", default=0.01, min=0.0, max=0.1, tooltip="Weight decay for regularization."),
                io.Boolean.Input("enable_lm_lora", default=True, tooltip="Apply LoRA to the Language Model backbone (q/k/v/o projections)."),
                io.Boolean.Input("enable_dit_lora", default=True, tooltip="Apply LoRA to the Diffusion Transformer (q/k/v/o projections)."),
                io.Boolean.Input("enable_proj_lora", default=False, tooltip="Apply LoRA to projection layers. Official default: disabled."),
                io.Combo.Input("lr_scheduler", options=["cosine", "constant"], default="cosine", tooltip="LR schedule: 'cosine' decays to 0 over training, 'constant' keeps LR fixed after warmup."),
            ],
            outputs=[
                VoxCPMV2TrainConfig.Output(display_name="V2 Train Config"),
            ],
        )

    @classmethod
    def execute(cls, **kwargs):
        kwargs["_version"] = "v2"
        return io.NodeOutput(kwargs)
