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
    if config['MODEL']['type'] == 'resnet':
        model = generate_resnet(
            number=config['MODEL']['resnet_version'], 
            pretrained=config['MODEL']['pretrained'], 
            current_device=current_device
        )
    elif config['MODEL']['type'] == 'baseline':
        model = BaselineModel().to(current_device)
    elif config['MODEL']['type'] == 'mobilenet':
        model = MobileNet().to(current_device)
    else:
        raise ValueError(f"Unknown model type: {config['MODEL']['type']}")
    return model


if __name__ == "__main__":

    # Load the configuration
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    
    # Configuration variables
    DATASET_PATH = config['DATASET_PATH']
    DATASET_NAME = config['DATASET_NAME']
    LEARNING_RATE = config['LEARNING_RATE']
    NUM_EPOCHS = config['NUM_EPOCHS']
    LOG_INTERVAL = config['LOG_INTERVAL']
    BATCH_SIZES = config['BATCH_SIZES']
    SPLIT_RATIOS = config['SPLIT_RATIOS']
    EARLY_STOPPER_CONFIG = config['EARLY_STOPPER']
    
    # Device selection: use GPU if available
    CURRENT_DEVICE = device("cuda:0" if is_available() else "cpu")
    print(f"Device Selected: {CURRENT_DEVICE}")

    # Fix random seed for reproducibility
    fix_random(42)

    # Model selection based on config
    MODEL = select_model(config, CURRENT_DEVICE)
    OPTIMIZER = Adam(MODEL.parameters(), lr=LEARNING_RATE)

    # Early stopping setup
    EARLY_STOPPER = EarlyStopper(
        patience=EARLY_STOPPER_CONFIG['patience'], 
        min_delta=EARLY_STOPPER_CONFIG['min_delta'],
    )

    # Define data transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        transforms.Resize([128, 128]),
    ])

    dataset = DatasetClass(
        extraction_path = DATASET_PATH, 
        dataset_name = DATASET_NAME, 
        train_batchsize = 256,
        eval_batchsize = 512, 
        split_dimension = SPLIT_RATIOS,
        transform = data_transforms,
    )

    # Display dataset distribution
    print(
        f"Dataset Distribution | "
        f"No Point: [{dataset.dataset['full'].targets.count(0)}] - "
        f"Point: [{dataset.dataset['full'].targets.count(1)}]\n"
        f"Split Distribution | "
        f"Train: [{len(dataset.dataset['train'])}] - "
        f"Validation: [{len(dataset.dataset['val'])}]"
    )

    # Calculate class weights for imbalance handling
    class_counts = [dataset.dataset['full'].targets.count(0), dataset.dataset['full'].targets.count(1)]
    total_count = sum(class_counts)
    weights = tensor([total_count / class_counts[0], total_count / class_counts[1]]).to(CURRENT_DEVICE)
    
    # Use weighted loss function
    loss_func = nn.BCEWithLogitsLoss(pos_weight=weights[1])

    # Training loop with early stopping
    history = training_loop(
        num_epochs=NUM_EPOCHS,
        optimizer=OPTIMIZER,
        log_interval=LOG_INTERVAL,
        model=MODEL,
        loader_train=dataset.dataloader['train'],
        loader_val=dataset.dataloader['val'],
        loss_func=loss_func,
        current_device=CURRENT_DEVICE,
        early_stopping=EARLY_STOPPER,
    )

    # Display training history
    display_history(history)

    # Make predictions and calculate metrics on the test set
    predictions, true_values, confidences = get_predictions(
        model=MODEL,
        data_loader=dataset.dataloader['val'],
        current_device=CURRENT_DEVICE,
    )
    metrics = calculate_metrics(predictions, true_values, confidences)

    # Print metrics
    for key, value in metrics.items():
        print(f"{key}: {value}")

    # Save the model with a unique filename including the F1 score
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/{MODEL.__class__.__name__}_{DATASET_NAME}_{metrics['f1_score']:.4f}_{timestamp}.pth'
    save(MODEL.state_dict(), model_path)