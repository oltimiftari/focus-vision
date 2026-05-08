import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from sklearn.model_selection import train_test_split


IMAGE_SIZE = 224
BATCH_SIZE = 8


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])


mask_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


class SaliencyDataset(Dataset):

    def __init__(self, image_dir, mask_dir, image_files,
                 image_transform=None, mask_transform=None):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = image_files

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):

        image_name = self.image_files[idx]

        image_path = os.path.join(self.image_dir, image_name)

        mask_name = image_name.replace(".jpg", ".png")
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


def create_dataloaders(image_dir, mask_dir):

    image_files = sorted(os.listdir(image_dir))

    train_images, val_images = train_test_split(
        image_files,
        test_size=0.2,
        random_state=42
    )

    train_dataset = SaliencyDataset(
        image_dir,
        mask_dir,
        train_images,
        image_transform=train_transform,
        mask_transform=mask_transform
    )

    val_dataset = SaliencyDataset(
        image_dir,
        mask_dir,
        val_images,
        image_transform=train_transform,
        mask_transform=mask_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader