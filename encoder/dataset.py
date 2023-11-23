import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Normalize, ToTensor, Compose


class RGBDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.img_paths = os.listdir(img_dir)
        self.transform = transform
        self.transform = Compose(
            [
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = read_image(img_path)
        if self.transform:
            image = image.float()
            image = self.transform(image)
        return image


class RGBDSegDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_dir = img_dir
        self.img_paths = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = read_image(img_path)
        if self.transform:
            image = self.transform(image)

        return image
