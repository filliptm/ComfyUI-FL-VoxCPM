"""Drop-in replacements for ``torchaudio.save`` / ``torchaudio.load`` that
do not require system FFmpeg shared libraries.

Modern torchaudio (>= 2.x) routes ``save``/``load`` through ``torchcodec``,
which loads ``libavutil.so.*`` / ``libavcodec.so.*`` etc. via ``ctypes.CDLL``.
On systems without system FFmpeg installed (e.g. minimal Linux containers),
this fails with ``OSError: libavutil.so.NN: cannot open shared object file``.

These helpers use ``soundfile`` (libsndfile via cffi), which is already a
project dependency and does not depend on FFmpeg.

Shape conventions match torchaudio:
    - ``save_audio`` accepts ``(channels, samples)`` torch tensors.
    - ``load_audio`` returns ``(waveform: (channels, samples), sample_rate)``.
"""

from __future__ import annotations

from typing import Tuple, Union
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


def save_audio(
    path: Union[str, Path],
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int,
) -> None:
    """Write a waveform to a WAV/FLAC/OGG file.

    Equivalent to ``torchaudio.save(path, waveform, sample_rate)`` but uses
    ``soundfile.write`` so no FFmpeg shared libraries are needed.
    """
    if isinstance(waveform, torch.Tensor):
        data = waveform.detach().cpu().numpy()
    else:
        data = np.asarray(waveform)
    # torchaudio uses (channels, samples); soundfile expects (samples,) for
    # mono or (samples, channels) for multichannel.
    if data.ndim == 2:
        data = data.T
    sf.write(str(path), data, int(sample_rate))


def load_audio(
    path: Union[str, Path],
) -> Tuple[torch.Tensor, int]:
    """Read an audio file into a torch tensor.

    Equivalent to ``torchaudio.load(path)`` returning
    ``(waveform: (channels, samples), sample_rate)``.
    """
    data, sr = sf.read(str(path), always_2d=False)
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        waveform = torch.from_numpy(arr).unsqueeze(0)
    else:
        # soundfile returns (samples, channels); torchaudio expects (channels, samples)
        waveform = torch.from_numpy(arr.T.copy())
    return waveform, int(sr)
