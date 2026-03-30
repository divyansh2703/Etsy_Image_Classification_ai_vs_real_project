import torch
from tqdm import tqdm
import numpy as np


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    preds = []
    probs = []
    targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Valid", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()

            prob_class_1 = torch.softmax(outputs, dim=1)[:, 1]
            pred_class = torch.argmax(outputs, dim=1)

            probs.extend(prob_class_1.cpu().numpy())
            preds.extend(pred_class.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return (
        total_loss / len(loader),
        np.array(preds),
        np.array(probs),
        np.array(targets),
    )