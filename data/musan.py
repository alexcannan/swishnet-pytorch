import os
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


MUSAN_ROOT_DIR = Path.home().joinpath("data", "MUSAN", "musan")  # Modify based on your dataset location


class MUSANDataset(Dataset):
    def __init__(self, root_dir: Path=MUSAN_ROOT_DIR, target_sample_rate: int=16000):
        """
        Args:
            root_dir (string): Directory with all the audio files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            target_sample_rate (int, optional): Target sample rate for audio files.
        """
        self.root_dir = root_dir
        self.target_sample_rate = target_sample_rate
        self.file_list = self._get_files()

    def _get_files(self) -> list[str]:
        file_list = []
        class_count: dict[str, int] = {}
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):  # Modify based on your dataset file types
                    file_list.append(os.path.join(root, file))
                    class_name = Path(root).parts[-2]
                    class_count[class_name] = class_count.get(class_name, 0) + 1
        self.class_count = class_count
        return file_list

    def resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx) -> tuple[torch.Tensor, str]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.file_list[idx]
        signal, sr = torchaudio.load(file_path)
        signal = self.resample_if_necessary(signal, sr)

        return signal, Path(file_path).parts[-3]


if __name__ == "__main__":
    import sys
    # Instantiate the dataset
    musan_dataset = MUSANDataset()

    # Access an example
    signal, file_path = musan_dataset[0]
    print(file_path, signal.shape)
    print(musan_dataset.class_count, file=sys.stderr)