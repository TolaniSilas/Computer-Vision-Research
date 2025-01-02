import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms_v2
from torchvision import datasets


torch.manual_seed(2024)

def load_MNIST_dataset():


    # Define the transformations to be performed on the training data.
    transformations = transforms_v2.Compose([
        # Resize the images.
        transforms_v2.Resize((224, 224)),

        # Random horizontal flip with a probability of 0.5.
        transforms_v2.RandomHorizontalFlip(p=0.5),

        # Random rotation of the image by a given angle (degrees).
        transforms_v2.RandomRotation(degrees=30),

        # Convert images to tensors.
        transforms_v2.ToTensor(),

        # Normalize the pixel values (in RGB channels).
        transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    

    # Define the transformations to be performed on the validation data.
    val_transformations = transforms_v2.Compose([
        # Resize the images.
        transforms_v2.Resize((224, 224)),

        # Convert images to tensors.
        transforms_v2.ToTensor(),

        # Normalize the pixel values (in RGB channels).
        transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])



    # Load the Fashion MNIST training image dataset and perform transformations.
    train_data = datasets.FashionMNIST(
        root="./data",
        train=True,
      transform=transformations,
      download=True
        )

    # Load the Fashion MNIST validation image dataset and perform transformations.
    valid_data = datasets.FashionMNIST(
        root="./data",
        train=False,
        transform=val_transformations,
        download=True
        )

    # Create a DataLoader for the training and validation dataset.
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_data, batch_size=128, shuffle=False, num_workers=0)

    return train_loader, valid_loader


print("Data are loaded and are ready to use!")