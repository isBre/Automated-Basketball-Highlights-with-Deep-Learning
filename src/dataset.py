import os
import zipfile
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import ImageFolder


class DatasetClass:
    """
    Represents a dataset with options for accessing different splits 
    (full, train, val, test) and their corresponding dataloaders.

    Usage Example:
    --------------
    Instantiate the class:
        ds = DatasetClass(args ...)
    
    Access datasets:
        ds['DATASET-NAME'].dataset['full']   # Full dataset
        ds['DATASET-NAME'].dataset['train']  # Training dataset (if enabled during initialization)
        ds['DATASET-NAME'].dataset['val']    # Validation dataset (if enabled during initialization)
        ds['DATASET-NAME'].dataset['test']   # Test dataset (if enabled during initialization)
    
    Access dataloaders:
        ds['DATASET-NAME'].dataloader['full']   # Full dataloader
        ds['DATASET-NAME'].dataloader['train']  # Training dataloader (if enabled during initialization)
        ds['DATASET-NAME'].dataloader['val']    # Validation dataloader (if enabled during initialization)
        ds['DATASET-NAME'].dataloader['test']   # Test dataloader (if enabled during initialization)
    
    Notes:
    ------
    - The availability of 'train', 'val', and 'test' splits depends on 
      the initialization parameters provided when creating an instance.
    - Datasets and dataloaders are stored in dictionaries, accessible via the dataset name.
    """
    
    def __init__(
        self, 
        extraction_path: str, 
        dataset_name: str,  
        train_batchsize: int,
        eval_batchsize: int, 
        split_dimension: list,
        transform: torchvision.transforms,
    ) -> None:
        """
        Initialize the DatasetClass
        
        Args:
            extraction_path: directory where to extract the dataset
            dataset_name: database name (folder name)
            train_batchsize: train batch size
            eval_batchsize: test and val batch size
            split_dimension: split proportions, you can chose between:
            [1] -> no split at all, only the 'full' option in the dictionaries will be available
            [x, y] -> one split, 'full', 'train', 'val' option are available in the dictionaries
            [x, y, z] -> two splits, 'full', 'train', 'val' and 'test' option are available in the dictionaries
            transform: the transform to apply to the data
        """
        # Construct the dataset_path
        # TODO this "/content" was on google colab
        self.dataset_path = os.path.join("/content", dataset_name)

        # Extract the dataset in the content folder
        # TODO same as before
        with zipfile.ZipFile(extraction_path, 'r') as zip_ref:
            zip_ref.extractall("/content")

        # Assign the transform and get the number of split
        self.transform = transform
        self.n_splits = len(split_dimension)

        self.dataset = {}
        # Generate the entire dataset ImageFolder 
        self.dataset['full'] = ImageFolder(self.dataset_path, transform = self.transform)

        # If we ask for a split, the class will create train and validation split
        if self.n_splits == 2:
            splits = random_split(self.dataset['full'], split_dimension)
            self.dataset['train'], self.dataset['val'] = splits[0], splits[1]
        
        #If we ask for two split, the class will create train, validation and test splits
        if self.n_splits == 3:
            splits = random_split(self.dataset['full'], split_dimension)
            self.dataset['train'], self.dataset['val'], self.dataset['test'] = splits[0], splits[1], splits[2]

        num_workers = os.cpu_count()

        #Assign batchsize
        self.train_batchsize = train_batchsize
        self.eval_batchsize = eval_batchsize

        self.dataloader = {}

        #Create the full dataloader of the dataset (it might be useful for experiments)
        self.dataloader['full'] = DataLoader(
            self.dataset['full'], 
            batch_size=self.train_batchsize,
            shuffle=False,
            num_workers=num_workers,
        )

        #If we ask for a split, the class will create train and validation dataloaders
        if self.n_splits > 1:
            self.dataloader['train'] = DataLoader(
                self.dataset['train'], 
                batch_size=self.train_batchsize,
                shuffle=True, 
                pin_memory=True,
                num_workers=num_workers,
            )
            
            self.dataloader['val'] = DataLoader(
                self.dataset['val'],
                batch_size=self.eval_batchsize,
                shuffle=False,
                num_workers=num_workers,
            )
        
        #If we ask for two split, the class will create train, validation and test dataloaders
        if self.n_splits > 2:
            self.dataloader['test'] = DataLoader(
                self.dataset['test'],
                batch_size=self.eval_batchsize,
                shuffle=False,
                num_workers=num_workers,
            )