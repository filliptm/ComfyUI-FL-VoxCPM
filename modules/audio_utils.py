"""Audio utility functions for VoxCPM nodes."""

import logging
import torch
import numpy as np
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


def comfyui_audio_to_tensor(audio: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    """Extract tensor and sample rate from ComfyUI audio format."""
    return audio["waveform"], audio["sample_rate"]


def ensure_mono(waveform: torch.Tensor) -> torch.Tensor:
    """Ensure audio is mono by averaging channels if stereo."""
    if waveform.dim() == 1:
        return waveform
    elif waveform.dim() == 2:
        if waveform.shape[0] > 1:
            return waveform.mean(dim=0)
        return waveform[0]
    elif waveform.dim() == 3:
        if waveform.shape[1] > 1:
            return waveform.mean(dim=1, keepdim=True)
        return waveform
    return waveform


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Resample audio to a target sample rate."""
    if orig_sr == target_sr:
        return waveform
    try:
        import torchaudio.transforms as T
        resampler = T.Resample(orig_sr, target_sr)
        return resampler(waveform)
    except ImportError:
        import librosa
        audio_np = waveform.numpy()
        if audio_np.ndim > 1:
            resampled = np.stack(
                [librosa.resample(ch, orig_sr=orig_sr, target_sr=target_sr) for ch in audio_np]
            )
        else:
            resampled = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=target_sr)
        return torch.from_numpy(resampled)
