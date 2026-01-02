import os
import numpy as np
import torch
from torch.utils.data import Dataset
import logging

log = logging.getLogger(__name__)


def load_data(filename, data_type, base_path):
    """
    Unified data loading function to reduce duplicated code
    and ensure correct path handling.
    """
    full_path = os.path.join(base_path, data_type, f"{filename}.npy")
    try:
        record = np.load(full_path)
        return torch.from_numpy(record).float()
    except FileNotFoundError:
        log.error(f"File not found: {full_path}")
        raise


class HSWIRFDatasetNewMix(Dataset):
    def __init__(self, file_path='./dataset/HS-WIRF_dataset/', is_train=True, scene='dml'):
        super(HSWIRFDatasetNewMix, self).__init__()
        self.is_train = is_train

        data_folder = 'train_data' if is_train else 'val_data'
        self.base_path = os.path.join(file_path, data_folder)

        # Use 'train' or 'val' based on is_train
        file_name = f'{scene}_{data_folder[:-5]}.txt'
        self.file = os.path.join(file_path, file_name)

        with open(self.file, 'r') as file:
            val_list = file.readlines()

        self.data = {'file_name': [], 'label': []}
        for line in val_list:
            parts = line.strip().split(',')
            self.data['file_name'].append(parts[0])
            self.data['label'].append(int(parts[2]) - 1)

        log.info("Loaded XRF dataset")

    def __len__(self):
        return len(self.data['label'])

    def __getitem__(self, idx):
        file_name = self.data['file_name'][idx]
        label = self.data['label'][idx]
        vector = self.word_list[label]

        wifi_data = load_data(file_name, 'WiFi', self.base_path)
        rfid_data = load_data(file_name, 'rfid', self.base_path)

        return wifi_data, rfid_data, label
