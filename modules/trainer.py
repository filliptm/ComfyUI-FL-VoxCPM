import os
import sys
import io as _io
import json
import base64
import torch
import torch.nn as nn
import logging
import contextlib
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

import comfy.model_management as model_management
from comfy.utils import ProgressBar

from ..src.voxcpm.model import VoxCPMModel
from ..src.voxcpm.model.voxcpm import LoRAConfig
from ..src.voxcpm.training import (
    Accelerator,
    BatchProcessor,
    build_dataloader,
    load_audio_text_datasets,
)
from .model_info import AVAILABLE_VOXCPM_MODELS

try:
    from safetensors.torch import save_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    from server import PromptServer
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

logger = logging.getLogger(__name__)


def send_training_update(node_id, data):
    if WEBSOCKET_AVAILABLE and PromptServer.instance is not None and node_id is not None:
        PromptServer.instance.send_sync(
            "voxcpm.training.progress",
            {"node": str(node_id), **data},
        )

def resolve_model_path(base_model_name: str, folder_paths_module):
    """Resolves the pretrained path for a given model name, downloading if official and missing."""
    model_info = AVAILABLE_VOXCPM_MODELS.get(base_model_name)
    if not model_info:
        raise ValueError(f"Base model '{base_model_name}' not found.")
    
    if model_info["type"] == "official":
        base_tts_path = os.path.join(folder_paths_module.get_folder_paths("tts")[0])
        voxcpm_models_dir = os.path.join(base_tts_path, "VoxCPM")
        pretrained_path = os.path.join(voxcpm_models_dir, base_model_name)
        
        if not os.path.exists(os.path.join(pretrained_path, "config.json")):
             from huggingface_hub import snapshot_download
             logger.info(f"Downloading official model {base_model_name}...")
             snapshot_download(repo_id=model_info["repo_id"], local_dir=pretrained_path)
    else:
        pretrained_path = model_info["path"]
        
    return pretrained_path

def _run_validation_audio(model, validation_text, validation_steps, sample_rate, save_dir, step, node_id):
    """Generate a validation audio sample and return base64-encoded WAV."""
    import torchaudio
    was_training = model.training
    device = next(model.parameters()).device
    original_dtype = next(model.parameters()).dtype

    # Temporarily move VAE to GPU for decoding
    vae_was_on_cpu = False
    if hasattr(model, 'audio_vae') and model.audio_vae is not None:
        vae_device = next(model.audio_vae.parameters()).device
        if vae_device != device:
            model.audio_vae.to(device)
            vae_was_on_cpu = True
    else:
        logger.warning(f"Validation skipped at step {step}: audio_vae not available on model.")
        return None

    model.eval()
    try:
        # Convert entire model to float32 for inference.
        # generate() internally creates tensors with dtype=float32, so the model
        # must also be float32 to avoid dtype mismatches.
        model.to(torch.float32)
        for param in model.parameters():
            param.data = param.data.float()
        for buf in model.buffers():
            buf.data = buf.data.float()
        # MiniCPM4's RotaryEmbedding reassigns cos_cached/sin_cached as plain tensors
        # (overwriting registered buffers), so we must convert them explicitly.
        for module in model.modules():
            for attr_name in ('cos_cached', 'sin_cached', 'inv_freq'):
                if hasattr(module, attr_name):
                    val = getattr(module, attr_name)
                    if isinstance(val, torch.Tensor) and val.is_floating_point():
                        setattr(module, attr_name, val.float())

        # Override model's config dtype to match the float32 inference
        original_config_dtype = getattr(model.config, 'dtype', None)
        if hasattr(model, 'config') and hasattr(model.config, 'dtype'):
            model.config.dtype = "float32"

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False), torch.amp.autocast('cpu', enabled=False):
            wav = model.generate(
                target_text=validation_text,
                inference_timesteps=validation_steps,
                cfg_value=2.0,
            )

        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0)
        save_path = os.path.join(save_dir, f"validation_step_{step}.wav")
        torchaudio.save(save_path, wav_tensor, sample_rate)

        buf = _io.BytesIO()
        torchaudio.save(buf, wav_tensor, sample_rate, format="wav")
        audio_b64 = f"data:audio/wav;base64,{base64.b64encode(buf.getvalue()).decode()}"
        return audio_b64
    except Exception as e:
        logger.warning(f"Validation audio generation failed at step {step}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Restore original dtype and training state
        if original_config_dtype is not None and hasattr(model, 'config') and hasattr(model.config, 'dtype'):
            model.config.dtype = original_config_dtype
        model.to(original_dtype)
        for param in model.parameters():
            param.data = param.data.to(original_dtype)
        for buf in model.buffers():
            buf.data = buf.data.to(original_dtype)
        for module in model.modules():
            for attr_name in ('cos_cached', 'sin_cached', 'inv_freq'):
                if hasattr(module, attr_name):
                    val = getattr(module, attr_name)
                    if isinstance(val, torch.Tensor) and val.is_floating_point():
                        setattr(module, attr_name, val.to(original_dtype))
        if was_training:
            model.train()
        # Move VAE back to CPU to free VRAM
        if vae_was_on_cpu and hasattr(model, 'audio_vae') and model.audio_vae is not None:
            model.audio_vae.to(device='cpu', dtype=torch.float32)


@torch.inference_mode(False)
def run_lora_training(
    base_model_name: str,
    train_config: dict,
    dataset_path: str,
    output_dir: str,
    max_steps: int,
    save_every_steps: int,
    num_workers: int,
    output_name: str,
    folder_paths_module,
    node_id=None,
    validation_text=None,
    validation_steps=10,
):
    """
    Executes the LoRA training loop.
    """

    torch.set_grad_enabled(True)

    logger.info(f"Training Environment Check - Grad Enabled: {torch.is_grad_enabled()}, Inference Mode: {torch.is_inference_mode_enabled()}")

    send_training_update(node_id, {"type": "status", "message": "Resolving model path..."})

    pretrained_path = resolve_model_path(base_model_name, folder_paths_module)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train_config.json"), 'w') as f:
        json.dump({**train_config, "base_model": base_model_name}, f, indent=2)

    model_management.unload_all_models()
    model_management.soft_empty_cache()

    accelerator = Accelerator(amp=True)

    lora_cfg = LoRAConfig(
        enable_lm=train_config.get("enable_lm_lora", True),
        enable_dit=train_config.get("enable_dit_lora", True),
        enable_proj=train_config.get("enable_proj_lora", False),
        r=train_config.get("lora_rank", 32),
        alpha=train_config.get("lora_alpha", 16),
        dropout=train_config.get("lora_dropout", 0.0),
    )

    send_training_update(node_id, {"type": "status", "message": "Loading base model..."})
    logger.info(f"Loading base model from {pretrained_path}...")
    base_model = VoxCPMModel.from_local(
        pretrained_path, 
        optimize=False, 
        training=True, 
        lora_config=lora_cfg
    )
    
    base_model = base_model.to(accelerator.device)

    trainable_params = [n for n, p in base_model.named_parameters() if p.requires_grad]
    logger.info(f"Detected {len(trainable_params)} trainable parameters.")
    
    if len(trainable_params) == 0:
        logger.warning("No trainable parameters detected! Forcing unfreeze of LoRA layers...")
        for name, param in base_model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        trainable_params = [n for n, p in base_model.named_parameters() if p.requires_grad]
        logger.info(f"After fix: {len(trainable_params)} trainable parameters.")
        if len(trainable_params) == 0:
            raise RuntimeError("Failed to enable gradients. Training aborted.")

    tokenizer = base_model.text_tokenizer

    send_training_update(node_id, {"type": "status", "message": "Loading dataset..."})
    logger.info("Loading dataset...")
    train_ds, _ = load_audio_text_datasets(
        train_manifest=dataset_path,
        sample_rate=train_config.get("sample_rate", 44100),
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    
    batch_size = 1 
    
    train_loader = build_dataloader(
        train_ds,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
    )

    dataset_cnt = int(max(train_ds["dataset_id"])) + 1 if "dataset_id" in train_ds.column_names else 1
    
    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_cnt,
        device=accelerator.device,
    )

    # Keep VAE on CPU for validation audio generation instead of deleting it
    if validation_text:
        audio_vae_ref = base_model.audio_vae.cpu()
        base_model.audio_vae = audio_vae_ref
    else:
        del base_model.audio_vae
    
    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()

    optimizer = AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=train_config.get("learning_rate", 1e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config.get("warmup_steps", 100),
        num_training_steps=max_steps,
    )

    send_training_update(node_id, {"type": "status", "message": "Starting training..."})
    logger.info("Starting training...")
    pbar = ProgressBar(max_steps)
    pbar.update(0)

    train_iter = iter(train_loader)
    data_epoch = 0
    grad_accum_steps = train_config.get("grad_accum_steps", 1)
    lambdas = {"loss/diff": 1.0, "loss/stop": 1.0}
    loss_history = []

    def get_next_batch():
        nonlocal train_iter, data_epoch
        try:
            return next(train_iter)
        except StopIteration:
            data_epoch += 1
            sampler = getattr(train_loader, 'sampler', None)
            if hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(data_epoch)
            train_iter = iter(train_loader)
            return next(train_iter)

    with torch.enable_grad():
        for step in range(max_steps):
            model_management.throw_exception_if_processing_interrupted()
            
            optimizer.zero_grad(set_to_none=True)
            
            total_loss_val = 0.0
            did_backward = False
            
            for micro_step in range(grad_accum_steps):
                batch = get_next_batch()
                processed = batch_processor(batch)
                
                is_last = (micro_step == grad_accum_steps - 1)
                sync_context = contextlib.nullcontext() if is_last else accelerator.no_sync()
                
                with sync_context:
                    with accelerator.autocast(dtype=torch.bfloat16):
                        outputs = model(
                            processed["text_tokens"],
                            processed["text_mask"],
                            processed["audio_feats"],
                            processed["audio_mask"],
                            processed["loss_mask"],
                            processed["position_ids"],
                            processed["labels"],
                            progress=step / max(1, max_steps),
                        )
                    
                    total_loss = 0.0
                    for key, value in outputs.items():
                        if key.startswith("loss/"):
                            weight = lambdas.get(key, 1.0)
                            if value.numel() > 1:
                                value = value.mean()
                            loss_value = value * weight / grad_accum_steps
                            total_loss = total_loss + loss_value
                    
                    if total_loss.grad_fn is not None:
                        accelerator.backward(total_loss)
                        total_loss_val += total_loss.item() * grad_accum_steps
                        did_backward = True
                    else:
                        if step == 0:
                            logger.error(f"Step 0: Loss has no grad_fn! Loss requires_grad: {total_loss.requires_grad}")
                        logger.warning(f"Step {step}: Skipping backward (no grad_fn)")

            if did_backward:
                scaler = getattr(accelerator, "scaler", None)
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=1.0)
                
                accelerator.step(optimizer)
                accelerator.update()
                scheduler.step()
                
                pbar.update(1)
            
            lr_val = optimizer.param_groups[0]['lr']
            logger.info(f"Step {step}/{max_steps}, Loss: {total_loss_val:.4f}, LR: {lr_val:.8f}")
            loss_history.append({"step": step, "loss": total_loss_val, "lr": lr_val})
            send_training_update(node_id, {
                "type": "progress",
                "step": step,
                "max_steps": max_steps,
                "loss": total_loss_val,
                "lr": lr_val,
                "loss_history": loss_history,
            })

            if (step + 1) % save_every_steps == 0 or (step + 1) == max_steps:
                save_path = os.path.join(output_dir, f"{output_name}_step_{step+1}.safetensors")
                
                full_state = unwrapped_model.state_dict()
                lora_state = {k: v for k, v in full_state.items() if "lora_" in k}
                
                if SAFETENSORS_AVAILABLE:
                    save_file(lora_state, save_path)
                else:
                    torch.save({"state_dict": lora_state}, save_path.replace(".safetensors", ".ckpt"))
                
                lora_info = {
                    "base_model": pretrained_path,
                    "lora_config": lora_cfg.model_dump() if hasattr(lora_cfg, "model_dump") else vars(lora_cfg),
                }
                config_path = os.path.join(output_dir, "lora_config.json")
                with open(config_path, "w", encoding="utf-8") as f:
                    json.dump(lora_info, f, indent=2)
                
                logger.info(f"Saved checkpoint: {save_path}")

                # Validation audio
                ckpt_update = {
                    "type": "checkpoint",
                    "step": step + 1,
                    "checkpoint_path": save_path,
                }
                if validation_text:
                    audio_b64 = _run_validation_audio(
                        unwrapped_model, validation_text, validation_steps,
                        train_config.get("sample_rate", 44100),
                        output_dir, step + 1, node_id
                    )
                    if audio_b64:
                        ckpt_update["validation_audio"] = audio_b64
                send_training_update(node_id, ckpt_update)

    del model, optimizer, scheduler, train_loader, base_model
    model_management.soft_empty_cache()

    send_training_update(node_id, {
        "type": "complete",
        "final_path": output_dir,
        "total_steps": max_steps,
    })

    return output_dir