import os

import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import torch


class MyDataset(Dataset):
    def __init__(self, dataset_path, use_rff = False):
        self.max_value = 0.0
        self.min_value = 0.0
        if not use_rff:
            self.data, self.labels = self.load_data(dataset_path)
        else:
            self.data, self.labels = self.load_rff(dataset_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        sample = (sample - self.min_value) / (self.max_value - self.min_value)
        return sample, label

    def load_data(self, dataset_path):
        data = []
        labels = []
        for device_id, device_folder in enumerate(os.listdir(dataset_path), 1):
            device_path = os.path.join(dataset_path, device_folder)
            if os.path.isdir(device_path):
                for file_name in os.listdir(device_path):
                    if file_name.endswith(".mat"):
                        file_path = os.path.join(device_path, file_name)
                        sample_data = loadmat(file_path)
                        I = torch.tensor(sample_data['I'])
                        Q = torch.tensor(sample_data['Q'])
                        sample = torch.cat([I.unsqueeze(0), Q.unsqueeze(0)], dim=0)
                        sample = sample.squeeze(-1)
                        
                        if torch.max(sample) > self.max_value:
                            self.max_value = torch.max(sample)
                        if torch.min(sample) < self.min_value:
                            self.min_value = torch.min(sample)
                        
                        data.append(sample)
                        labels.append(device_id - 1)
        return data, labels

    def load_rff(self, dataset_path):
        data = []
        labels = []
        for device_id, device_folder in enumerate(os.listdir(dataset_path), 1):
            device_path = os.path.join(dataset_path, device_folder)
            if os.path.isdir(device_path):
                for file_name in os.listdir(device_path):
                    if file_name.endswith(".npy"):
                        file_path = os.path.join(device_path, file_name)
                        sample = np.load(file_path)
                        sample = torch.tensor(sample)
                        if torch.max(sample) > self.max_value:
                            self.max_value = torch.max(sample)
                        if torch.min(sample) < self.min_value:
                            self.min_value = torch.min(sample)
                        data.append(sample)
                        labels.append(device_id - 1)
        return data, labels



