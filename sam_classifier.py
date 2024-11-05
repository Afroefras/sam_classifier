import torch
import torchaudio
from pathlib import Path
from typing import List, Tuple, Dict
from helpers import audio_utils as AU
from pytorch_lightning import LightningModule


class SAM_Dataset(torch.utils.data.Dataset):
    def __init__(self, base_dir: Path, chunk_secs: float, transform=None) -> None:
        self.base_dir = base_dir
        self.transform = transform
        self._load_data(chunk_secs)
        self._create_label_mapping()

    def _create_label_mapping(self) -> None:
        audio_labels = set(name for _, name in self.data)
        self.label_to_idx = {name: idx for idx, name in enumerate(sorted(audio_labels))}
        self.idx_to_label = {idx: name for name, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.data)

    def make_heartbeats_chunks(
        self, audio: torch.Tensor, sample_rate: int, chunk_secs: float
    ) -> List[torch.Tensor]:
        chunk_size = int(sample_rate * chunk_secs)
        chunks = torch.split(audio, chunk_size, dim=-1)
        chunks = list(chunks)

        if chunk_size > chunks[-1].shape[-1]:
            chunks.pop(-1)

        return chunks

    def _load_data(self, chunk_secs: float) -> None:
        self.data = []
        self.data_dir = list(self.base_dir.glob("*.wav"))

        for path in self.data_dir:
            audio_label = path.stem
            audio, sample_rate = torchaudio.load(str(path))
            audio_chunks = self.make_heartbeats_chunks(audio, sample_rate, chunk_secs)

            n_audios = len(audio_chunks)
            names = [audio_label] * n_audios
            chunks = list(zip(audio_chunks, names))

            self.data.extend(chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        audio, audio_label = self.data[idx]
        label = self.label_to_idx[audio_label]

        if self.transform:
            sample = {"audio": audio, "audio_label": audio_label}
            sample = self.transform(sample)

            audio = sample["audio"]

            try:
                mel_spec = sample["mel_spec"]
            except KeyError:
                mel_spec = [None]

            label = self.label_to_idx[sample["audio_label"]]

        return audio, mel_spec, label


class ResampleAudio(object):
    def __init__(self, orig_sample_rate: int, new_sample_rate: int) -> None:
        self.orig_sample_rate = orig_sample_rate
        self.new_sample_rate = new_sample_rate
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.orig_sample_rate, new_freq=self.new_sample_rate
        )

    def __call__(self, sample) -> Dict[str, torch.Tensor]:
        audio, audio_label = sample["audio"], sample["audio_label"]
        audio_resampled = self.resampler(audio)
        return {"audio": audio_resampled, "audio_label": audio_label}


class LowpassFilter(object):
    def __init__(self, sample_rate: int, cutoff_freq: int, order: int) -> None:
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.order = order

    def __call__(self, sample) -> Dict[torch.Tensor, int]:
        audio, audio_label = sample["audio"], sample["audio_label"]

        audio_filtered = AU.apply_lowpass_filter(
            audio,
            sample_rate=self.sample_rate,
            cutoff_freq=self.cutoff_freq,
            order=self.order,
        )

        return {"audio": audio_filtered, "audio_label": audio_label}


class Normalize(object):
    def __call__(self, sample) -> Dict[torch.Tensor, int]:
        audio, audio_label = sample["audio"], sample["audio_label"]
        audio_scaled = AU.standard_scale(audio)
        return {"audio": audio_scaled, "audio_label": audio_label}


class ToMelSpectrogram(object):
    def __init__(
        self, sample_rate: int, n_mels: int, n_fft: int, hop_length: int
    ) -> None:
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, sample) -> Dict:
        audio, audio_label = sample["audio"], sample["audio_label"]

        mel_spec_db = AU.compute_mel_spectrogram(
            audio,
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        return {
            "audio": audio,
            "mel_spec": torch.Tensor(mel_spec_db),
            "audio_label": audio_label,
        }


class Compose:
    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class SAM_CNN_Model(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.pool1 = torch.nn.MaxPool2d(2)
        self.dropout1 = torch.nn.Dropout(0.25)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool2 = torch.nn.MaxPool2d(2)
        self.dropout2 = torch.nn.Dropout(0.25)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.pool3 = torch.nn.MaxPool2d(2)
        self.dropout3 = torch.nn.Dropout(0.25)

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(256)
        self.pool4 = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout4 = torch.nn.Dropout(0.25)

        self.fc1 = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.pool1(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = self.pool3(torch.nn.functional.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        x = self.pool4(torch.nn.functional.relu(self.bn4(self.conv4(x))))
        x = self.dropout4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _, x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)
        self.log("val_loss", loss, on_epoch=True)
        self.log("val_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0005, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }
