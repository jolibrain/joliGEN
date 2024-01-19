"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""


import os
import os.path

import torch
import torch.nn.functional as F
from torch.fft import fft, ifft
import torch.utils.data as data

# TODO optional?
import torchaudio


def window(t):
    """
    t between 0 & 1
    """
    return (1 - torch.cos(t * (torch.pi * 2))) / 2


# TODO write a test to check that `wav2D_to_wav(wav_to_2D(x))` is consistent
def wav_to_2D(data, chunk_size, norm_factor):
    """
    Transform sound data to image-like data (2D, normalized between -1 & 1)
    """
    chunk_gap = chunk_size // 2
    chunks = torch.stack(
        [
            data[i : i + chunk_size]
            for i in range(0, len(data) - chunk_size + 1, chunk_gap)
        ]
    )
    chunks_fft = fft(chunks)[:, : chunk_size // 2]
    chunks_fft = torch.stack([chunks_fft.real, chunks_fft.imag, torch.abs(chunks_fft)])
    chunks_fft /= norm_factor
    # TODO manage sound longer than input size
    # TODO don't hard code input size
    chunks_fft = torch.nn.functional.pad(
        chunks_fft, (0, 0, 0, 256 - chunks_fft.shape[-2]), value=0
    )
    # print(torch.max(chunks_fft), torch.min(chunks_fft))
    return chunks_fft


def wav2D_to_wav(sound2d, norm_factor):
    """
    Transform image-like data (2D, normalized between -1 & 1) to waveform. This
    function is the inverse of wav_to_2D

    Parameters:
        sound2d -- The 2D matrix containing the sound, with shape [n_channel, width, height]
    """
    # sound2d: channel, time, fourier features
    chunk_size = sound2d.shape[-1] * 2
    sound2d = (sound2d[0] + 1j * sound2d[1]) * norm_factor
    chunks_fft = F.pad(sound2d, (0, chunk_size // 2), mode="constant", value=0)
    chunks = ifft(chunks_fft).real

    # Apply window and paste chunks together
    chunk_window = window(torch.linspace(0, 1, chunk_size + 1, device=sound2d.device))[
        :-1
    ]
    chunks = chunks * chunk_window

    chunks_odd = F.pad(torch.flatten(chunks[1::2]), (chunk_size // 2, 0))
    chunks_even = torch.flatten(chunks[0::2])
    total_size = min(len(chunks_even), len(chunks_odd))

    signal = chunks_odd[:total_size] + chunks_even[:total_size]
    return signal.unsqueeze(0)


def load_sound(sound_path):
    data, rate = torchaudio.load(sound_path)

    # Ensure mono audio
    data = data[0]

    # TODO dynamic chunk_size
    chunk_size = 512
    norm_factor = 256  # 65536
    return wav_to_2D(data, chunk_size, norm_factor)
