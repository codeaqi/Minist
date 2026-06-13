import time

import torch


class Accuracy:
    def __call__(self, outputs, labels):
        _, predictions = torch.max(outputs, 1)
        return (predictions == labels).float().mean().item()


def evaluate(model, data_loader, dataset_size, device):
    model.eval()
    correct = 0
    last_images = None
    last_predictions = None
    start = time.time()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum()
            last_images = images
            last_predictions = predictions

    elapsed = time.time() - start
    accuracy = correct.cpu().numpy() / dataset_size * 100
    return accuracy, elapsed, last_images, last_predictions
