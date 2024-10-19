import torch.nn as nn
from typing import Dict, List
from torch.utils.data import DataLoader
from torch import device, no_grad, squeeze
from sklearn.metrics import (accuracy_score, precision_score, 
                             recall_score, f1_score, 
                             confusion_matrix, roc_auc_score)


def get_predictions(
        model: nn.Module, 
        dataloader: DataLoader,
        current_device: device,
        threshold: float = 0.5,
    ) -> Dict[str, List]:
    """
    Evaluate a given model using a dataloader and return prediction results.

    Args:
        model (nn.Module): The model to be evaluated.
        dataloader (DataLoader): The dataloader containing the dataset to evaluate the model on.
        current_device (torch.device): The device on which the model and data will be processed (e.g., "cpu" or "cuda").
        threshold (float, optional): The threshold for converting confidence scores to binary predictions. 
                                     Default is 0.5 (values greater than 0.5 will be predicted as 1, otherwise 0).

    Returns:
        Dict: A dictionary containing the following:
            - 'confidences' (List[float]): A list of confidence scores (float values between 0 and 1).
            - 'predictions' (List[int]): A list of predicted binary values (1 if confidence > threshold, 0 otherwise).
            - 'true_values' (List[int]): A list of true binary labels (0 or 1).
    """
    confidences = []
    true_values = []

    # Set model to evaluation mode (disables dropout and batchnorm layers)
    model.eval()

    # Context-manager that disabled gradient calculation
    with no_grad():

        # Iterate through the dataloader in batches
        for images, labels in dataloader:
            # Move images and labels to the specified device (CPU or GPU)
            images, labels = images.to(current_device), labels.to(current_device)

            # Forward pass: Get model predictions
            # Squeeze the outputs to match expected dimension (batch_size,)
            outputs = squeeze(model(images))

            # Accumulate confidences and true labels
            confidences.extend(outputs.tolist())
            true_values.extend(labels.tolist())

    # Generate binary predictions based on the confidence threshold
    predictions = [1 if x > threshold else 0 for x in confidences]

    return {
        "confidences": confidences,
        "predictions": predictions,
        "true_values": true_values
    }


def calculate_metrics(predictions, true_values, confidences=None):
    accuracy = accuracy_score(true_values, predictions)
    precision = precision_score(true_values, predictions)
    recall = recall_score(true_values, predictions)
    f1 = f1_score(true_values, predictions)
    conf_matrix = confusion_matrix(true_values, predictions)
    roc_auc = roc_auc_score(true_values, confidences) if confidences is not None else None
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "roc_auc": roc_auc
    }