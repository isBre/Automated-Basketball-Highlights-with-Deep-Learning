import numpy as np
import torch.nn as nn
from typing import Callable, Tuple
from torch.utils.data import DataLoader
from torch import Tensor, device, optim, squeeze, float32, no_grad


class EarlyStopper:
    """
    Represent an object that is able to tell when the train need to stop.
    Needs to be put inside the train and keep updated, when early_stop returns
    true we need to stop the train
    """

    def __init__(self, patience: int=1, min_delta: int=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_correct_samples(scores: Tensor, labels: Tensor) -> int:
    """Get the number of correctly classified examples.

    Args:
        scores: the probability distribution.
        labels: the class labels.

    Returns: :return: the number of correct samples
    """
    classes_predicted = (scores > 0.5).float()
    return (classes_predicted == labels).sum().item()


def train(
        model: nn.Module,
        train_loader: DataLoader,
        device: device,
        optimizer: optim,
        criterion: Callable[[Tensor, Tensor], float],
        log_interval: int,
        epoch: int
    ) -> Tuple[float, float]:
    """Train loop to train a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        device: the device to use to train the model.        
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.
        log_interval: the log interval.
        epoch: the number of the current epoch

    Returns:
        the Cross Entropy Loss value on the training data, 
        the accuracy on the training data.
    """
    # Corrected Labeled samples
    correct = 0

    # Images in the batch
    samples_train = 0

    # Loss of the Training Set
    loss_train = 0

    # Entire dimension of the Training Set
    size_ds_train = len(train_loader.dataset)

    # Number of batches
    num_batches = len(train_loader)

    # IMPORTANT: from now on, since we will introduce batch norm, we have to tell PyTorch if we are training or evaluating our model
    model.train()

    # Loop inside the train_loader
    # The batch size is definited inside the train_loader
    for idx_batch, (images, labels) in enumerate(train_loader):

      # In order to speed up the process I want to use the current device
      images, labels = images.to(device), labels.to(device)

      # Set the gradient of the available parameters to zero
      optimizer.zero_grad()

      # Get the output of the model
      # I need to squeeze because of the dimension of the output (x, 1), I want just (x)
      outputs = squeeze(model(images))

      # Here the model calculate the loss comparing true values and obtained values
      # Here i need to cast to float32 because: labels is long and outputs is float32
      loss = criterion(outputs, labels.to(float32))

      # Update the total loss adding the loss of this particular batch
      loss_train += loss.item() * len(images)

      # Update the number of analyzed images
      samples_train += len(images)
      
      # Compute the gradient
      loss.backward()

      # Update parameters considering the loss.backward() values
      optimizer.step()

      # Update the number of correct predicted values adding the correct value of this batch
      correct += get_correct_samples(outputs, labels)

      # Update metrics
      if log_interval > 0:
          if idx_batch % log_interval == 0:
              running_loss = loss_train / samples_train
              global_step = idx_batch + (epoch * num_batches)

    loss_train /= samples_train
    accuracy_training = 100. * correct / samples_train
    return loss_train, accuracy_training


def validate(
        model: nn.Module,
        data_loader: DataLoader,
        device: device,
        criterion: Callable[[Tensor, Tensor], float]) -> Tuple[float, float]:
    """Evaluate the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.
        device: the device to use to evaluate the model.
        criterion: the loss function.

    Returns:
        the loss value on the validation data 
        the accuracy on the validation data
    """
    # Corrected Labeled samples
    correct = 0

    # Images in the batch
    samples_val = 0

    # Loss of the Valuation Set
    loss_val = 0.

    # IMPORTANT: from now on, since we will introduce batch norm, we have to tell PyTorch if we are training or evaluating our model
    model = model.eval()

    # Context-manager that disabled gradient calculation
    with no_grad():

      # Loop inside the data_loader
      # The batch size is definited inside the data_loader
      for idx_batch, (images, labels) in enumerate(data_loader):

        # In order to speed up the process I want to use the current device
        images, labels = images.to(device), labels.to(device)

        # Get the output of the model
        # I need to squeeze because of the dimension of the output (x, 1), I want just (x)
        outputs = squeeze(model(images))

        # Here the model calculate the loss comparing true values and obtained values
        # Here i need to cast to float32 because: labels is long and outputs is float32
        loss = criterion(outputs, labels.to(float32))

        # Update metrics
        loss_val += loss.item() * len(images)
        samples_val += len(images)
        correct += get_correct_samples(outputs, labels)

    loss_val /= samples_val
    accuracy = 100. * correct / samples_val
    return loss_val, accuracy