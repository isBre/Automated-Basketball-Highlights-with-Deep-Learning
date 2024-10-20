"""
This dataset was created with the help of `YOLOv5`, previously trained to recognize baskets.
The dataset consists of small frames and ~7000 game screenshots that are precisely labeled.
  * Image size: 128x128
  * Total images: 6863 (644 labeled as `Point`, 6219 as `No Point`)
"""

import yaml
import torch.nn as nn
from torch.optim import Adam
from datetime import datetime
from src.utils import fix_random
from torchvision import transforms
from torch.cuda import is_available
from src.dataset import DatasetClass
from torch import device, tensor, save
from src.plotter import display_history
from src.models.mobile_net import MobileNet
from src.models.resnet import generate_resnet
from src.models.baseline_model import BaselineModel
from src.training import EarlyStopper, training_loop
from src.evaluation import calculate_metrics, get_predictions


def select_model(config, current_device):
    # TODO move this somewhere else
    """Dynamically select and initialize the model based on the YAML configuration."""
    if config['model']['type'] == 'resnet':
        model = generate_resnet(
            number=config['model']['resnet_version'], 
            pretrained=config['model']['pretrained'], 
            current_device=current_device
        )
    elif config['model']['type'] == 'baseline':
        model = BaselineModel().to(current_device)
    elif config['model']['type'] == 'mobilenet':
        model = MobileNet().to(current_device)
    else:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    return model


if __name__ == "__main__":

    # Load the configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Configuration variables
    dataset_paths = config['dataset_paths']
    learning_rate = config['learning_rate']
    n_epochs = config['n_epochs']
    log_interval = config['log_interval']
    split_ratios = config['split_ratios']
    batch_size = config['batch_size']
    early_stopper_config = config['early_stopper']
    
    # Device selection: use GPU if available
    current_device = device("cuda:0" if is_available() else "cpu")
    print(f"Device Selected: {current_device}")

    # Fix random seed for reproducibility
    fix_random(42)

    # Model selection based on config
    model = select_model(config, current_device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Early stopping setup
    early_stopper = EarlyStopper(
        patience=early_stopper_config['patience'], 
        min_delta=early_stopper_config['min_delta'],
    )

    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize([128, 128]),
    ])

    dataset = DatasetClass(
        folder_paths=dataset_paths, 
        batchsize=batch_size,
        split_ratios=split_ratios,
        transform=data_transforms,
    )

    # Display dataset distribution
    targets = []
    for d in dataset.full_dataset.datasets:
        targets.extend(d.targets)

    print(
        f"Dataset Distribution | "
        f"No Point: [{targets.count(0)}] - "
        f"Point: [{targets.count(1)}]\n"
        f"Split Distribution | "
        f"Train: [{len(dataset.datasets[0])}] - "  # Access the length of the first dataset (train)
        f"Validation: [{len(dataset.datasets[1])}]"  # Access the length of the second dataset (validation)
    )

    # Calculate class counts for imbalance handling
    class_counts = [targets.count(0), targets.count(1)]
    total_count = sum(class_counts)

    # Compute class weights
    weights = tensor([total_count / class_counts[0], total_count / class_counts[1]]).to(current_device)

    # Use weighted loss function
    loss_func = nn.BCEWithLogitsLoss(pos_weight=weights[1])

    # Training loop with early stopping
    history = training_loop(
        num_epochs=n_epochs,
        optimizer=optimizer,
        log_interval=log_interval,
        model=model,
        loader_train=dataset.dataloaders[0],
        loader_val=dataset.dataloaders[1],
        loss_func=loss_func,
        current_device=current_device,
        early_stopping=early_stopper,
    )

    # Display training history
    display_history(history)

    # Save the model with a unique filename including the F1 score
    timestamp = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    model_path = f'models/{model.__class__.__name__}_{timestamp}.pt'
    save(model.state_dict(), model_path)