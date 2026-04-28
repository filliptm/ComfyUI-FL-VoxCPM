import os
import torch
import logging
import tempfile
from typing import Optional

import folder_paths
import comfy.model_management as model_management
from comfy_api.latest import io, ui

from ..modules.model_info import AVAILABLE_VOXCPM_MODELS, V2_MODEL_NAMES
from ..modules.loader import VoxCPMModelHandler, detect_lora_rank
from ..modules.patcher import VoxCPMPatcher

logger = logging.getLogger(__name__)

VOXCPM_V2_PATCHER_CACHE = {}

V2_MODES = [
    "Text to Speech",
    "Voice Design",
    "Voice Cloning",
    "Controllable Cloning",
    "Ultimate Cloning",
]


def get_available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    try:
        import platform
        if platform.system() == "Windows" and hasattr(torch.backends, 'directml') and torch.backends.directml.is_available():
            devices.append("directml")
    except Exception:
        pass
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def set_seed(seed: int):
    if seed < 0:
        seed = torch.randint(0, 666666, (1,)).item()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _get_v2_model_names():
    return [name for name in AVAILABLE_VOXCPM_MODELS if name in V2_MODEL_NAMES]


def _get_voxcpm_lora_list():
    """Get LoRAs from the VoxCPM subdirectory."""
    loras = ["None"]
    lora_dirs = folder_paths.get_folder_paths("loras")
    for lora_dir in lora_dirs:
        voxcpm_dir = os.path.join(lora_dir, "VoxCPM")
        if os.path.isdir(voxcpm_dir):
            for root, dirs, files in os.walk(voxcpm_dir):
                for f in files:
                    if f.endswith((".safetensors", ".ckpt", ".pth")):
                        rel = os.path.relpath(os.path.join(root, f), lora_dir)
                        loras.append(rel)
    return loras


def _extract_audio(audio_input):
    """Extract waveform and sample_rate from ComfyUI audio dict."""
    if isinstance(audio_input, dict) and 'waveform' in audio_input and 'sample_rate' in audio_input:
        waveform = audio_input['waveform']
        if waveform.dim() == 3:
            waveform = waveform[0]
        if waveform.numel() == 0:
            raise ValueError("Provided audio is empty.")
        return waveform, audio_input['sample_rate']
    raise ValueError("Invalid audio format.")


def _save_waveform_to_temp(waveform, sample_rate):
    """Save a waveform tensor to a temporary WAV file."""
    from ..modules.audio_io import save_audio
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    save_audio(tmp.name, waveform.cpu(), sample_rate)
    tmp.close()
    return tmp.name


class FL_VoxCPM_V2_TTS(io.ComfyNode):
    CATEGORY = "FL/VoxCPM"

    @classmethod
    def define_schema(cls) -> io.Schema:
        v2_names = _get_v2_model_names()
        if not v2_names:
            v2_names = ["No V2 models found. Please download VoxCPM2."]

        available_devices = get_available_devices()
        lora_list = _get_voxcpm_lora_list()

        return io.Schema(
            node_id="FL_VoxCPM_V2_TTS",
            display_name="FL VoxCPM V2 TTS",
            category=cls.CATEGORY,
            is_output_node=True,
            description="Generate speech using VoxCPM V2 with Voice Design, Voice Cloning, Controllable Cloning, and Ultimate Cloning modes.",
            inputs=[
                io.Combo.Input("model_name", options=v2_names, default=v2_names[0], tooltip="Select a VoxCPM V2 model."),
                io.Combo.Input("lora_name", options=lora_list, default="None", tooltip="Select a LoRA from loras/VoxCPM/."),
                io.Combo.Input("mode", options=V2_MODES, default="Voice Design", tooltip="Generation mode."),
                io.String.Input("text", multiline=True, default="Hello, this is a test of VoxCPM V2.", tooltip="Text to synthesize."),
                io.String.Input("control", multiline=True, optional=True, default="", tooltip="Voice design instructions, e.g. 'young woman, gentle and sweet voice'. Used in Voice Design and Controllable Cloning modes."),
                io.Audio.Input("reference_audio", optional=True, tooltip="V2 voice identity reference audio. Used in Voice Cloning, Controllable Cloning, and Ultimate Cloning."),
                io.String.Input("reference_text", multiline=True, optional=True, default="", tooltip="Transcript of reference audio. Used in Ultimate Cloning mode."),
                io.Audio.Input("prompt_audio", optional=True, tooltip="Continuation audio. Used in Ultimate Cloning mode."),
                io.String.Input("prompt_text", multiline=True, optional=True, default="", tooltip="Transcript of prompt audio. Used in Ultimate Cloning mode."),
                io.Float.Input("cfg_value", default=2.0, min=1.0, max=10.0, step=0.1, tooltip="Guidance scale."),
                io.Int.Input("inference_timesteps", default=10, min=1, max=100, step=1, tooltip="Number of diffusion steps."),
                io.Int.Input("min_tokens", default=2, min=1, max=100, tooltip="Minimum audio token length."),
                io.Int.Input("max_tokens", default=2048, min=64, max=8192, tooltip="Maximum audio token length."),
                io.Boolean.Input("normalize_text", default=True, tooltip="Enable text normalization."),
                io.Int.Input("seed", default=-1, min=-1, max=0xFFFFFFFFFFFFFFFF, tooltip="Seed for reproducibility. -1 for random."),
                io.Boolean.Input("force_offload", default=False, tooltip="Force VRAM offload after generation."),
                io.Combo.Input("device", options=available_devices, default=available_devices[0], tooltip="Inference device."),
                io.Int.Input("retry_max_attempts", default=3, min=0, max=10, step=1, tooltip="Max retry attempts for bad output."),
                io.Float.Input("retry_threshold", default=6.0, min=2.0, max=20.0, step=0.1, tooltip="Audio/text ratio threshold for retry."),
            ],
            outputs=[
                io.Audio.Output(display_name="Generated Audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_name: str,
        lora_name: str,
        mode: str,
        device: str,
        text: str,
        cfg_value: float,
        inference_timesteps: int,
        min_tokens: int,
        max_tokens: int,
        normalize_text: bool,
        seed: int,
        force_offload: bool,
        retry_max_attempts: int,
        retry_threshold: float,
        control: Optional[str] = None,
        reference_audio: Optional[io.Audio.Type] = None,
        reference_text: Optional[str] = None,
        prompt_audio: Optional[io.Audio.Type] = None,
        prompt_text: Optional[str] = None,
    ) -> io.NodeOutput:

        # Guard: ensure this is a V2 model
        model_info = AVAILABLE_VOXCPM_MODELS.get(model_name)
        if model_name not in V2_MODEL_NAMES:
            if model_info:
                raise ValueError(f"'{model_name}' is a V1 model. Use the FL VoxCPM TTS node instead.")
            raise ValueError(f"Model '{model_name}' not found.")

        # Clean optional strings
        control = control.strip() if control else ""
        reference_text = reference_text.strip() if reference_text else ""
        prompt_text_str = prompt_text.strip() if prompt_text else ""

        # Validate mode requirements
        if mode == "Text to Speech":
            pass  # No requirements — just text, optional LoRA
        elif mode == "Voice Design":
            if not control:
                raise ValueError("Voice Design mode requires 'control' text describing the desired voice.")
            if reference_audio is not None or prompt_audio is not None:
                raise ValueError("Voice Design mode does not accept audio inputs. Remove reference/prompt audio.")
        elif mode == "Voice Cloning":
            if reference_audio is None:
                raise ValueError("Voice Cloning mode requires 'reference_audio'.")
        elif mode == "Controllable Cloning":
            if reference_audio is None:
                raise ValueError("Controllable Cloning mode requires 'reference_audio'.")
            if not control:
                raise ValueError("Controllable Cloning mode requires 'control' text for style instructions.")
        elif mode == "Ultimate Cloning":
            if reference_audio is None:
                raise ValueError("Ultimate Cloning requires 'reference_audio'.")
            if not reference_text:
                raise ValueError("Ultimate Cloning requires 'reference_text' (transcript of reference audio).")
            if prompt_audio is None:
                raise ValueError("Ultimate Cloning requires 'prompt_audio'.")
            if not prompt_text_str:
                raise ValueError("Ultimate Cloning requires 'prompt_text' (transcript of prompt audio).")

        # Build final text with control prefix (only for modes that use it)
        final_text = text
        if control and mode in ("Voice Design", "Controllable Cloning"):
            final_text = f"({control}){text}"

        # Device setup
        if device == "cuda":
            load_device = model_management.get_torch_device()
            offload_device = model_management.intermediate_device()
        else:
            load_device = torch.device("cpu")
            offload_device = torch.device("cpu")

        # LoRA setup
        lora_rank = 32
        lora_path = None
        if lora_name != "None":
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                raise FileNotFoundError(f"LoRA file not found: {lora_name}")
            lora_rank = detect_lora_rank(lora_path)

        # Load model
        cache_key = f"v2_{model_name}_{device}_r{lora_rank}"
        if cache_key not in VOXCPM_V2_PATCHER_CACHE:
            handler = VoxCPMModelHandler(model_name, lora_rank=lora_rank)
            patcher = VoxCPMPatcher(handler, load_device=load_device, offload_device=offload_device, size=handler.size)
            VOXCPM_V2_PATCHER_CACHE[cache_key] = patcher

        patcher = VOXCPM_V2_PATCHER_CACHE[cache_key]
        model_management.load_model_gpu(patcher)
        voxcpm_model = patcher.model.model

        if not voxcpm_model:
            raise RuntimeError(f"Failed to load VoxCPM V2 model '{model_name}'.")

        # Verify it's actually a V2 model
        if not getattr(voxcpm_model, 'is_v2', False):
            raise ValueError(f"'{model_name}' loaded as V1. This node requires a V2 model.")

        if lora_path:
            voxcpm_model.load_lora(lora_path)
            voxcpm_model.set_lora_enabled(True)
        else:
            voxcpm_model.set_lora_enabled(False)

        set_seed(seed)
        enable_retry = retry_max_attempts > 0
        temp_files = []

        try:
            # Prepare reference audio as temp file (V2 uses file paths)
            ref_wav_path = None
            if reference_audio is not None:
                ref_waveform, ref_sr = _extract_audio(reference_audio)
                ref_wav_path = _save_waveform_to_temp(ref_waveform, ref_sr)
                temp_files.append(ref_wav_path)

            # Prepare prompt audio
            prompt_waveform = None
            prompt_sample_rate = None
            prompt_wav_path = None
            if prompt_audio is not None:
                p_waveform, p_sr = _extract_audio(prompt_audio)
                prompt_wav_path = _save_waveform_to_temp(p_waveform, p_sr)
                temp_files.append(prompt_wav_path)

            wav_array = voxcpm_model.generate(
                text=final_text,
                prompt_text=prompt_text_str or None,
                prompt_wav_path=prompt_wav_path,
                reference_wav_path=ref_wav_path,
                cfg_value=cfg_value,
                inference_timesteps=inference_timesteps,
                min_len=min_tokens,
                max_len=max_tokens,
                normalize=normalize_text,
                retry_badcase=enable_retry,
                retry_badcase_max_times=max(retry_max_attempts, 1),
                retry_badcase_ratio_threshold=retry_threshold,
                denoise=False
            )

            output_tensor = torch.from_numpy(wav_array).float().unsqueeze(0).unsqueeze(0)
            output_sr = voxcpm_model.tts_model.sample_rate
            output_audio = {"waveform": output_tensor, "sample_rate": output_sr}

            if force_offload:
                patcher.unpatch_model(unpatch_weights=True)

            return io.NodeOutput(output_audio, ui=ui.PreviewAudio(output_audio, cls=cls))

        finally:
            for tmp_path in temp_files:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
