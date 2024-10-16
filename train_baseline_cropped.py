"""
I created this dataset with the help of `YoloV5` previously trailed in recognizing baskets 
and fed it with `Small-Frames` and other 7000 screnshoot of games precisely classified. 
  * Images size: 128x128
  * Images: 6863 (644 `Point`, 6219 `No Point`)
  * It will be divided into: train, validation, test (70, 15, 15)
"""

import torch.nn as nn
from torch import device
from torch.optim import Adam
from src.utils import fix_random
from torchvision import transforms
from torch.cuda import is_available
from src.dataset import DatasetClass
from src.baseline_model import BaselineModel
from src.training import EarlyStopper, training_loop


if __name__ == "__main__":
    
    DATASET_PATH = "datasets/Cropped.zip"
    DATASET_NAME = DATASET_PATH.split('/')[-1].split('.')[0]

    current_device = device("cuda:0" if is_available() else "cpu")
    print(f"Device Selected: {current_device}")
    fix_random(42)

    cr_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
        ),
        transforms.Resize([128, 128])])

    dataset = DatasetClass(
        extraction_path = DATASET_PATH, 
        dataset_name = DATASET_NAME, 
        train_batchsize = 256,
        eval_batchsize = 512, 
        split_dimension = [0.7, 0.15, 0.15],
        transform = cr_transform,
    )

    print(
        f'Dataset Distribution \n'
        f'Number of No Point images: {dataset.dataset["full"].targets.count(0)} \n'
        f'Number of Point images: {dataset.dataset["full"].targets.count(1)}'
    )
    print(
        f'Split Distribution: \n'
        f'Training Set: {len(dataset.dataset["train"])} images\n'
        f'Validation Set: {len(dataset.dataset["val"])} images\n'
        f'Test Set: {len(dataset.dataset["test"])} images'
    )

    lr = 0.001
    num_epochs = 500
    log_interval = 20

    baseline_model = BaselineModel()
    optimizer = Adam(baseline_model.parameters(), lr=lr)
    early_stopper = EarlyStopper(patience=5, min_delta=0)
    history = training_loop(
        num_epochs = num_epochs,
        optimizer = optimizer, 
        log_interval = log_interval,
        model = baseline_model,
        loader_train = dataset.dataloader['train'], 
        loader_val = dataset.dataloader['val'],
        loss_func = nn.BCELoss(),
        current_device = current_device,
        early_stopping = early_stopper,
    )
    print(history)