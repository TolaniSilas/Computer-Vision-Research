"""
create dataloaders for train and validation datasets.
these dataloaders will be to train the ViT model.
"""

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms_v2
from torchvision import datasets


torch.manual_seed(2024)

def load_stl10(data_dir: str, train_batch_size: int, val_batch_size: int):
    """loads stl10 dataset from the given directory and returns train and val dataloaders."""

    # define the transformations to be performed on the training data.
    train_transform = transforms_v2.Compose([
        # resize the images.
        transforms_v2.Resize((112, 112)),

        # random horizontal flip with a probability of 0.5.
        transforms_v2.RandomHorizontalFlip(p=0.5),

        # random vertical flip with a probability of 0.5.
        transforms_v2.RandomVerticalFlip(p=0.5),

        # convert images to tensors.
        transforms_v2.ToTensor(),

        # normalize the pixel values (in RGB channels).
        transforms_v2.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2241, 0.2215, 0.2239]
        )
    ])

    # define transforms for validation data.
    val_transform = transforms_v2.Compose([
        # resize the images.
        transforms_v2.Resize((112, 112)),

        # convert images to tensors.
        transforms_v2.ToTensor(),

        # normalize the pixel values (in RGB channels).
        transforms_v2.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2241, 0.2215, 0.2239]
        )
    ])

    # load train and validation datasets from directory.
    train_dataset = datasets.STL10(
        root=data_dir,
        split="train",
        transform=train_transform,
        download=False
    )

    val_dataset = datasets.STL10(
        root=data_dir,
        split="test",
        transform=val_transform,
        download=False
    )

    # create dataloaders for train and validation datasets.
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
    )

    return train_loader, val_loader



# train_loader, val_loader = load_stl10(
#     data_dir="./data",
#     train_batch_size=32,
#     val_batch_size=64,
# )