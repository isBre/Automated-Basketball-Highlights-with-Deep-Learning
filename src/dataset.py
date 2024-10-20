import os
from typing import List
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split, ConcatDataset


class DatasetClass:
    """
    A class to handle dataset creation, splitting, and data loading for training and evaluation.

    Attributes:
        folder_paths (List[str]): List of paths to image folders.
        train_batchsize (int): Batch size for training.
        eval_batchsize (int): Batch size for evaluation.
        split_ratios (List[float]): Ratios for splitting the dataset.
        transform (torchvision.transforms): Transformations to apply to images.
    """

    def __init__(
        self,
        folder_paths: List[str],
        batchsize: int,
        split_ratios: List[float],
        transform: T.Compose,
    ) -> None:
        self.folder_paths = folder_paths
        self.batchsize = batchsize
        self.split_ratios = split_ratios
        self.transform = transform

        self.full_dataset = self._create_full_dataset()
        self.datasets = self._split_dataset()
        self.full_dataloader = self._create_dataloader(self.full_dataset, batchsize, shuffle=False)
        self.dataloaders = self._create_dataloaders()

    def _create_full_dataset(self) -> ConcatDataset:
        """Creates a full dataset by concatenating datasets from all specified folders."""
        image_folders = [ImageFolder(root=path, transform=self.transform) for path in self.folder_paths]
        return ConcatDataset(image_folders)

    def _split_dataset(self) -> List[ConcatDataset]:
        """Splits the full dataset into smaller datasets based on the split ratios."""
        return random_split(self.full_dataset, self.split_ratios)

    def _create_dataloader(self, dataset, batch_size: int, shuffle: bool) -> DataLoader:
        """Creates a DataLoader for the given dataset."""
        num_workers = os.cpu_count()
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )

    def _create_dataloaders(self) -> List[DataLoader]:
        """Creates DataLoaders for each split dataset."""
        return [
            self._create_dataloader(dataset, self.batchsize, shuffle=True)
            for dataset in self.datasets
        ]
