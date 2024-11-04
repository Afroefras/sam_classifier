import torch
import numpy as np


def get_all_predictions(trained_model, dataloader, device=None):
    trained_model.eval()
    if device is None:
        device = next(trained_model.parameters()).device
    trained_model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            _, mel_spec, labels = batch

            mel_spec = mel_spec.to(device)
            labels = labels.to(device)

            input_data = mel_spec.unsqueeze(1)

            outputs = trained_model(input_data)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def compute_confusion_matrix(all_labels, all_preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(all_labels, all_preds):
        cm[true_label, pred_label] += 1

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    return cm, cm_normalized
