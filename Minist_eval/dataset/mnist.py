import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def _load_processed_dataset(data_root, split):
    processed_paths = {
        "train": os.path.join(data_root, "processed", "training", "training.pt"),
        "test": os.path.join(data_root, "processed", "test", "test.pt"),
    }
    data_path = processed_paths[split]
    if not os.path.exists(data_path):
        return None

    images, labels = torch.load(data_path, map_location="cpu")
    images = images.float().unsqueeze(1) / 255.0
    labels = labels.long()
    return TensorDataset(images, labels)


def build_mnist_loaders(data_root=".", batch_size=100, val_batch_size=None, download=True):
    val_batch_size = val_batch_size or batch_size
    transform = transforms.ToTensor()

    train_dataset = _load_processed_dataset(data_root, "train")
    test_dataset = _load_processed_dataset(data_root, "test")

    if train_dataset is None or test_dataset is None:
        train_dataset = datasets.MNIST(
            root=data_root,
            train=True,
            transform=transform,
            download=download,
        )
        test_dataset = datasets.MNIST(
            root=data_root,
            train=False,
            transform=transform,
            download=download,
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        shuffle=True,
        batch_size=val_batch_size,
    )

    return train_loader, test_loader, train_dataset, test_dataset
