import torch
import numpy as np
import seaborn as sns
from typing import Union
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from helpers import audio_utils as AU
from helpers.predictions import get_all_predictions, compute_confusion_matrix


def plot_waveform(audio: np.array, sample_rate: int, ax: plt.Axes) -> None:
    time_axis = np.arange(audio.shape[-1]) / sample_rate
    ax.plot(time_axis, audio)
    ax.set_title("Forma de onda")
    ax.set_ylabel("Amplitud")
    ax.set_xlabel("Tiempo (segundos)")


def plot_fourier_spectrogram(
    audio: np.array, sample_rate: int, ax: plt.Axes
) -> plt.Axes:
    f, t, Sxx = spectrogram(audio, fs=sample_rate, nperseg=1024, noverlap=512)
    im = ax.pcolormesh(
        t, f, 10 * np.log10(Sxx + 1e-10), shading="gouraud", cmap="Blues"
    )
    ax.set_title("Espectrograma de Fourier (STFT)")
    ax.set_ylabel("Frecuencia (Hz)")
    ax.set_xlabel("Tiempo (segundos)")
    return im


def plot_mel_spectrogram(
    audio: torch.Tensor,
    sample_rate: int,
    n_mels: int,
    n_fft: int,
    hop_length: int,
    ax: plt.Axes,
) -> plt.Axes:

    mel_spec_db = AU.compute_mel_spectrogram(
        audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )

    duration = audio.shape[-1] / sample_rate

    im = ax.imshow(
        mel_spec_db,
        aspect="auto",
        origin="lower",
        cmap="Blues",
        extent=[0, duration, 0, 128],
    )
    ax.set_title("Espectrograma de Mel")
    ax.set_ylabel("Mel bins")
    ax.set_xlabel("Tiempo (segundos)")
    return im


def plot_waveform_and_spectrograms(
    audio: Union[np.array, torch.Tensor],
    sample_rate: int,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 512,
) -> None:
    audio = np.reshape(audio, -1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    plot_waveform(audio, sample_rate, axs[0])

    im1 = plot_fourier_spectrogram(audio, sample_rate, axs[1])
    fig.colorbar(im1, ax=axs[1], orientation="horizontal", label="Intensidad (dB)")

    im2 = plot_mel_spectrogram(
        audio=audio,
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        ax=axs[2],
    )
    fig.colorbar(im2, ax=axs[2], orientation="horizontal", label="Intensidad (dB)")

    plt.tight_layout()
    plt.show()


def plot_audio_fft(audio: np.array, sample_rate: int) -> None:
    freq, magn = AU.get_positive_freq_and_magn(audio, sample_rate)

    plt.figure(figsize=(12, 3))
    plt.plot(freq, magn)
    plt.title("Distribución de Frecuencias")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud")
    plt.show()


def compare_audios(
    first_audio: np.array,
    first_title: str,
    second_audio: np.array,
    second_title: str,
    sample_rate: int,
) -> None:
    first_audio = first_audio.view(-1)
    second_audio = second_audio.view(-1)

    time_axis = np.linspace(0, len(second_audio) / sample_rate, num=len(second_audio))

    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    plt.plot(time_axis, first_audio, label=first_title, alpha=0.7, color="skyblue")
    plt.plot(time_axis, second_audio, label=second_title, alpha=0.7)
    plt.ylabel("Amplitud")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time_axis, first_audio, label=first_title, alpha=0.7, color="skyblue")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time_axis, second_audio, label=second_title, alpha=0.7)
    plt.xlabel("Tiempo (segundos)")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_model_result(trained_model, dataset, index):
    trained_model.eval()
    with torch.no_grad():
        audio, mel_spec, label = dataset[index]

        input_data = mel_spec.view([1, -1, mel_spec.shape[0], mel_spec.shape[1]])

        outputs = trained_model(input_data)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, 1)

        idx_to_label = dataset.dataset.idx_to_label
        actual_label_name = idx_to_label[int(label)]
        predicted_label_name = idx_to_label[int(predicted)]

        fig, axs = plt.subplots(2, 1, figsize=(12, 8))

        axs[0].plot(audio.squeeze().cpu().numpy())
        axs[0].set_title("Forma de onda del audio")
        axs[0].set_xlabel("Tiempo")
        axs[0].set_ylabel("Amplitud")

        axs[1].imshow(
            mel_spec.squeeze().cpu().numpy(),
            origin="lower",
            aspect="auto",
            cmap="Blues",
        )
        axs[1].set_title("Espectrograma de Mel")
        axs[1].set_xlabel("Tiempo")
        axs[1].set_ylabel("Frecuencia Mel")

        is_correct = actual_label_name == predicted_label_name
        suptitle = "Correcto!\n" if is_correct else "Incorrecto:\n"
        suptitle += (
            predicted_label_name
            if is_correct
            else f"Predicción del modelo: {predicted_label_name}"
        )
        suptitle += "" if is_correct else f"\nEtiqueta real: {actual_label_name}"

        plt.suptitle(suptitle, fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

        return probabilities


def plot_confusion_matrix(trained_model, dataloader):

    all_preds, all_labels = get_all_predictions(trained_model, dataloader)

    classes_dict = dataloader.dataset.dataset.idx_to_label
    n_classes = classes_dict.__len__()
    class_names = list(classes_dict.values())

    cm, cm_normalized = compute_confusion_matrix(all_labels, all_preds, n_classes)
    accuracy = np.trace(cm) / np.sum(cm) * 100

    flatten_cms = zip(cm.flatten(), cm_normalized.flatten())
    labels = np.array(
        [f"{value}\n{percentage:.1f}%" for value, percentage in flatten_cms]
    )
    labels = labels.reshape(cm.shape)

    annot_kws = {"color": "white", "fontsize": 10, "alpha": 1}

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws=annot_kws,
    )
    plt.ylabel("Etiqueta Real")
    plt.xlabel("Predicción")
    plt.title(f"Matriz de Confusión - Precisión total: {accuracy:.1f}%")
    plt.show()

    return all_preds, all_labels, cm
