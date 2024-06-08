import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image

class NCKUFinalDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.transform = transform
        self.img_labels = []

        with open(txt_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                path, label = line.strip().split()
                self.img_labels.append((path, int(label)))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label