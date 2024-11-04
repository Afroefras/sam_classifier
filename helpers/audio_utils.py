import torch
import torchaudio
import numpy as np
from scipy import signal
from pathlib import Path
from scipy.io import wavfile
from typing import Union, Tuple


def standard_scale(x: torch.Tensor) -> torch.Tensor:
    x_mean = x.mean()
    x_std = x.std() + 1e-10

    scaled = x.clone()
    scaled -= x_mean
    scaled /= x_std

    return scaled


def min_max_scale(x: torch.Tensor) -> torch.Tensor:
    x = torch.Tensor(x)

    x_min = x.min()
    x_max = x.max()

    scaled = x.clone()
    scaled -= x_min
    scaled /= x_max - x_min

    return scaled


def get_positive_freq_and_magn(
    audio: np.array, sample_rate: int
) -> Tuple[np.array, np.array]:
    audio = audio.squeeze()

    fft_result = np.fft.fft(audio)
    fft_magnitude = np.abs(fft_result)
    frequencies = np.fft.fftfreq(len(fft_result), 1 / sample_rate)

    positive_frequencies = frequencies[frequencies >= 0]
    positive_fft_magnitude = fft_magnitude[: len(positive_frequencies)]

    return positive_frequencies, positive_fft_magnitude


def apply_bandpass_filter(
    waveform: torch.Tensor, sample_rate: int, low_freq: int, high_freq: int
) -> torch.Tensor:

    central_freq = (low_freq + high_freq) / 2
    bandwidth = high_freq - low_freq
    Q = central_freq / bandwidth

    filtered_waveform = torchaudio.functional.bandpass_biquad(
        waveform, sample_rate, central_freq, Q
    )
    return filtered_waveform


def apply_lowpass_filter(
    waveform: torch.Tensor, sample_rate: int, cutoff_freq: int, order: int
) -> torch.Tensor:
    nyquist = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist
    b, a = signal.butter(order, normalized_cutoff, btype="low", analog=False)

    filtered_waveform = signal.lfilter(b, a, waveform.numpy())

    return torch.Tensor(filtered_waveform)


def trim_audio(
    audio: torch.Tensor, sample_rate: int, start_at: float = None, end_at: float = None
):
    if start_at is None:
        start_at = 0
    if end_at is None:
        end_at = audio.shape[-1] // sample_rate

    starts = int(start_at * sample_rate)
    ends = int(end_at * sample_rate)

    return audio[:, starts:ends]


def resample_audio(
    audio: torch.Tensor, sample_rate: int, new_sample_rate: int
) -> Tuple[torch.Tensor, int]:
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=new_sample_rate
    )
    audio = resampler(audio)
    sample_rate = new_sample_rate
    return audio, sample_rate


def add_noise(
    audio: torch.Tensor, noise: torch.Tensor, noise_vol: float
) -> torch.Tensor:
    if audio.shape[-1] > noise.shape[-1]:
        repeat_factor = audio.shape[-1] // noise.shape[-1] + 1
        noise = noise.repeat(1, repeat_factor)

    noise = noise[:, : audio.shape[-1]]

    audio_noisy = noise * noise_vol + audio * (1 - noise_vol)
    return audio_noisy


def compute_mel_spectrogram(
    audio: Union[np.array, torch.torch.Tensor],
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 512,
    n_mels: int = 128,
) -> np.array:

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    mel_spec_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_spec = mel_spec_transform(audio)
    mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec_db = mel_spec_db.squeeze().numpy()
    return mel_spec_db


def save_wave_to_wav(
    wave: np.ndarray,
    sample_rate: int,
    filedir: Path,
    filename: str,
    volume: float = 1.0,
) -> None:
    wave = min_max_scale(wave.astype(np.float32))
    wave = wave * volume

    wave = np.clip(wave, -1.0, 1.0)
    wave_int16 = np.int16(wave * 32767)

    filename += ".wav"
    file = filedir.joinpath(filename)

    wavfile.write(str(file), sample_rate, wave_int16)
    print(f"El archivo '{filename}' se ha guardado exitosamente en:\n{filedir}")
