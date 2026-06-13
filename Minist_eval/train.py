import argparse
import os
import sys

import torch
import torchvision
from torchvision.utils import make_grid

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Minist_eval.dataset.mnist import build_mnist_loaders
from Minist_eval.utils.config import first_component_config, instantiate_from_config, load_yaml_config
from Minist_eval.utils.device import get_device
from Minist_eval.utils.metrics import evaluate
from Minist_eval.utils.visualize import imshow, setup_chinese_font


DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "v1.0_baseline_Scratch.yaml")


def train(model, train_loader, device, cfg):
    hp = cfg["hyperparameters"]
    epochs = int(hp.get("epochs_num") or hp.get("epochs") or 10)
    loss_fn = instantiate_from_config(first_component_config(hp["loss"]))
    optimizer = instantiate_from_config(hp["optimizer"], params=model.parameters())

    scheduler = None
    if hp.get("lr_scheduler"):
        scheduler = instantiate_from_config(hp["lr_scheduler"], optimizer=optimizer)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        if scheduler is not None:
            scheduler.step()

        accuracy = correct / total * 100
        avg_loss = running_loss / len(train_loader)
        print("Epoch [{}/{}] loss: {:.4f}, acc: {:.2f}%".format(epoch + 1, epochs, avg_loss, accuracy))


def load_weights(model, model_path, device):
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device)


def save_weights(model, model_path):
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print("Model saved to {}".format(model_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Train or evaluate MNIST CNN.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to yaml config.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    hp = cfg["hyperparameters"]

    setup_chinese_font()
    device = get_device(bool(hp.get("use_gpu", True)))
    print("Using config: {}".format(args.config))
    print("Using device: {}".format(device))

    train_batch_size = int(hp.get("global_batch_size") or 100)
    val_batch_size = int(hp.get("val_batch_size") or train_batch_size)
    data_root = hp.get("data_root") or "."
    download = bool(hp.get("download", True))
    show_pic = bool(hp.get("show_pic", False))
    is_train = bool(hp.get("is_train", False))
    model_path = hp.get("model_path") or hp.get("pretrained_weights") or os.path.join(".", "modelpara.pth")

    train_loader, test_loader, _, test_dataset = build_mnist_loaders(
        data_root=data_root,
        batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        download=download,
    )

    if show_pic:
        train_images, _ = next(iter(train_loader))
        imshow(make_grid(train_images, nrow=10, padding=2, pad_value=1), "Train samples")

    model = instantiate_from_config(hp["model"]).to(device)

    if is_train:
        train(model, train_loader, device, cfg)
        save_weights(model, model_path)
    else:
        model = load_weights(model, model_path, device)

    if show_pic:
        test_images, _ = next(iter(test_loader))
        imshow(torchvision.utils.make_grid(test_images, nrow=10, padding=2, pad_value=1), "Test samples")

    accuracy, elapsed, last_images, last_predictions = evaluate(
        model=model,
        data_loader=test_loader,
        dataset_size=len(test_dataset),
        device=device,
    )

    if show_pic and last_images is not None and last_predictions is not None:
        imshow(
            torchvision.utils.make_grid(last_images[75:100].cpu(), nrow=5, padding=2, pad_value=1),
            "25 test predictions:\n" + str(last_predictions[75:100].cpu().numpy()),
        )

    print("MNIST test accuracy = {:.2f}%".format(accuracy))
    print("10000 images inference time: {:.3f} s".format(elapsed))


if __name__ == "__main__":
    main()
