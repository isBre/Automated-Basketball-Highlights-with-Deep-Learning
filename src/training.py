import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Callable, Tuple, Dict
from timeit import default_timer as timer
from sklearn.metrics import f1_score, accuracy_score
from torch import Tensor, optim, squeeze, float32, no_grad, device


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


def train(
        model: nn.Module,
        train_loader: DataLoader,
        current_device: device,
        optimizer: optim,
        criterion: Callable[[Tensor, Tensor], float],
        log_interval: int,
        epoch: int,
    ) -> Tuple[float, float]:
    """Train loop to train a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        current_device: the device to use to train the model.        
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.
        log_interval: the log interval.
        epoch: the number of the current epoch

    Returns:
        the Cross Entropy Loss value on the training data, 
        the accuracy on the training data,
        the F1 score on the training data.
    """
    # Images in the batch
    samples_train = 0

    # Loss of the Training Set
    loss_train = 0

    # Lists to store true labels and predicted outputs for accuracy and F1 score calculation
    all_labels = []
    all_predictions = []

    # Set the model to training mode
    model.train()

    # Loop inside the train_loader
    for idx_batch, (images, labels) in enumerate(train_loader):

        # Move data to the current device
        images, labels = images.to(current_device), labels.to(current_device)

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

        # Convert model outputs to predictions (0 or 1 based on threshold)
        predicted = (outputs > 0.5).float()

        # Store predictions and labels for accuracy and F1 score calculation
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate average loss
    loss_train /= samples_train
    
    # Calculate accuracy using sklearn
    accuracy_training = accuracy_score(all_labels, all_predictions)

    # Calculate F1 score using sklearn
    f1_training = f1_score(all_labels, all_predictions, average='binary')  # Use 'binary' for binary classification

    return loss_train, accuracy_training, f1_training


def validate(
        model: nn.Module,
        data_loader: DataLoader,
        current_device: device,
        criterion: Callable[[Tensor, Tensor], float],
    ) -> Tuple[float, float, float]:
    """Evaluate the model.

    Args:
        model: the model to evaluate.
        data_loader: the data loader containing the validation or test data.
        current_device: the device to use to evaluate the model.
        criterion: the loss function.

    Returns:
        the loss value on the validation data 
        the accuracy on the validation data
        the F1 score on the validation data
    """
    # Images in the batch
    samples_val = 0

    # Loss of the Validation Set
    loss_val = 0.

    # Lists to store true labels and predicted outputs for accuracy and F1 score calculation
    all_labels = []
    all_predictions = []

    # Set the model to evaluation mode
    model = model.eval()

    # Context-manager that disables gradient calculation
    with no_grad():

        # Loop inside the data_loader
        for idx_batch, (images, labels) in enumerate(data_loader):

            # Move data to the current device
            images, labels = images.to(current_device), labels.to(current_device)

            # Get the output of the model
            outputs = squeeze(model(images))

            # Here the model calculate the loss comparing true values and obtained values
            # Here i need to cast to float32 because: labels is long and outputs is float32
            loss = criterion(outputs, labels.to(float32))

            # Update loss metric
            loss_val += loss.item() * len(images)
            samples_val += len(images)

            # Convert model outputs to predictions (0 or 1 based on threshold)
            predicted = (outputs > 0.5).float()

            # Store predictions and labels for accuracy and F1 score calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate average loss
    loss_val /= samples_val

    # Calculate accuracy using sklearn
    accuracy = accuracy_score(all_labels, all_predictions)

    # Calculate F1 score
    f1 = f1_score(all_labels, all_predictions, average='binary')  # Use 'binary' for binary classification

    return loss_val, accuracy, f1


def training_loop(
        num_epochs: int,
        optimizer: optim,
        log_interval: int, 
        model: nn.Module, 
        loader_train: DataLoader, 
        loader_val: DataLoader,
        loss_func: nn.modules.loss,
        current_device: device,
        verbose: bool = True,
        early_stopping: EarlyStopper = None
    ) -> Dict:
    """Executes the training loop.
    
        Args:
            name_exp: the name for the experiment.
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            log_interval: intervall to print on tensorboard.
            model: the mode to train.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            verbose: 

        Returns:  
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch
            the values for the train accuracy for each epoch
            the values for the validation accuracy for each epoch
            the time of execution in seconds for the entire loop
        """

    # Represent the Loss Function
    criterion = loss_func

    # Start the timer in order to obtain the time needed to entirely train the model
    loop_start = timer()

    # Record the history
    train_losses_values = []
    val_losses_values = []
    train_acc_values = []
    val_acc_values = []
    train_f1_values = []
    val_f1_values = []

    # For every epoch
    for epoch in range(1, num_epochs + 1):

        # Start the timer in order to obtain the time needed to train in this epoch
        time_start = timer()

        # Obtain Loss and Accuracy for the train step
        loss_train, accuracy_train, f1_train = train(
            model=model, 
            train_loader=loader_train, 
            current_device=current_device,
            optimizer=optimizer, 
            criterion=criterion, 
            log_interval=log_interval,
            epoch=epoch,
        )
        
        # Obtain Loss and Accuracy from the validation step
        loss_val, accuracy_val, f1_val = validate(
            model=model, 
            data_loader=loader_val, 
            current_device=current_device, 
            criterion=criterion,
        )

        if early_stopping is not None:
            if early_stopping.early_stop(loss_val):
                print(f'--- Early Stopping ---')     
                break

        #Stop the timer for this step
        time_end = timer()

        # Update history
        train_losses_values.append(loss_train)
        val_losses_values.append(loss_val)
        train_acc_values.append(accuracy_train)
        val_acc_values.append(accuracy_val)
        train_f1_values.append(f1_train)
        val_f1_values.append(f1_val)
        
        # Stop the timer for the entire training
        loop_end = timer()

        # Calculate total time
        time_loop = loop_end - loop_start

        # Metrics Print
        lr =  optimizer.param_groups[0]['lr']
        if verbose:            
            print(
                f"Epoch: [{epoch}] | Lr: [{lr:.4f}] | "
                f"Train Loss: [{loss_train:.4f}] | Val Loss: [{loss_val:.4f}] | "
                f"Train Acc: [{accuracy_train:.4f}] | Val Acc: [{accuracy_val:.4f}] | "
                f"Train F1: [{f1_train:.4f}] | Val F1: [{f1_val:.2f}] | "
                f"Time/Epoch (s): {(time_end - time_start):.4f} | Total Time (s): {time_loop:.2f}"
            )
            
    return {
        'train_loss_values': train_losses_values,
        'val_loss_values' : val_losses_values,
        'train_acc_values': train_acc_values,
        'val_acc_values': val_acc_values,
        'time': time_loop
    }