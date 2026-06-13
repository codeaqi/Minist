import torch


def get_device(use_gpu=False):
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

