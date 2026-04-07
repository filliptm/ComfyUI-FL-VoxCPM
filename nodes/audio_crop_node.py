"""Audio crop/trim node for VoxCPM."""

import logging
import torch
from typing import Tuple, Dict, Any

logger = logging.getLogger(__name__)


class FL_VoxCPM_AudioCrop:
    """Crop (trim) audio to a specific start and end time."""

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "crop_audio"
    CATEGORY = "FL/VoxCPM"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", {"description": "Input audio tensor"}),
                "start_time": ("STRING", {
                    "default": "0:00",
                    "description": "Start time (MM:SS or seconds)"
                }),
                "end_time": ("STRING", {
                    "default": "1:00",
                    "description": "End time (MM:SS or seconds)"
                }),
            }
        }

    def crop_audio(self, audio: Dict[str, Any], start_time: str = "0:00", end_time: str = "1:00") -> Tuple[Dict[str, Any]]:
        try:
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            if ":" not in start_time:
                start_time = f"00:{start_time}"
            if ":" not in end_time:
                end_time = f"00:{end_time}"

            start_seconds = 60 * int(start_time.split(":")[0]) + int(start_time.split(":")[1])
            start_frame = start_seconds * sample_rate

            end_seconds = 60 * int(end_time.split(":")[0]) + int(end_time.split(":")[1])
            end_frame = end_seconds * sample_rate

            total_frames = waveform.shape[-1]
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))

            if start_frame >= end_frame:
                raise ValueError(
                    f"Start time ({start_time}) must be less than end time ({end_time}) "
                    f"and be within the audio length."
                )

            cropped_waveform = waveform[..., start_frame:end_frame]

            cropped_audio = {
                'waveform': cropped_waveform,
                'sample_rate': sample_rate
            }

            duration = (end_frame - start_frame) / sample_rate
            logger.info(f"Audio cropped: {start_time} to {end_time} ({duration:.2f}s)")

            return (cropped_audio,)

        except Exception as e:
            logger.error(f"Audio crop failed: {e}")
            import traceback
            traceback.print_exc()
            return (audio,)
