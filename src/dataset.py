import torchvision
from torchvision.datasets import ImageFolder
import zipfile
import torch
import os


class DatasetClass:
  """
  A class representing the dataset under consideration. 
  Splits in the dataset and dataloader will be considered as dictionaries,
  the user can then use the following wording to obtain the desired split:

  ds = DatasetClass(args ...)

  ds['DATASET-NAME'].dataset['full'] <- obtain the full dataset
  ds['DATASET-NAME'].dataset['train'] <- obtain the train dataset (if allowed in the initilization)
  ds['DATASET-NAME'].dataset['val'] <- obtain the val dataset (if allowed in the initilization)
  ds['DATASET-NAME'].dataset['test'] <- obtain the val dataset (if allowed in the initilization)

  ds['DATASET-NAME'].dataloader['full'] <- obtain the full dataloader
  ds['DATASET-NAME'].dataloader['train'] <- obtain the train dataloader (if allowed in the initilization)
  ds['DATASET-NAME'].dataloader['val'] <- obtain the val dataloader (if allowed in the initilization)
  ds['DATASET-NAME'].dataloader['test'] <- obtain the test dataloader (if allowed in the initilization)
  """

  def __init__(self, 
               extraction_path : str, 
               dataset_name : str,  
               train_batchsize : int,
               eval_batchsize : int, 
               split_dimension : list,
               transform : torchvision.transforms):
  
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

    #Construct the dataset_path
    self.dataset_path = "/content/" + dataset_name

    #Extract the dataset in the content folder
    with zipfile.ZipFile(extraction_path, 'r') as zip_ref:
      zip_ref.extractall("/content")

    #Assign the transform and get the number of split
    self.transform = transform
    self.n_splits = len(split_dimension)

    self.dataset = {}
    #generate the entire dataset ImageFolder 
    self.dataset['full'] = ImageFolder(self.dataset_path, transform = self.transform)

    #If we ask for a split, the class will create train and validation split
    if self.n_splits == 2:
      splits = torch.utils.data.random_split(self.dataset['full'], split_dimension)
      self.dataset['train'], self.dataset['val'] = splits[0], splits[1]
    
    #If we ask for two split, the class will create train, validation and test splits
    if self.n_splits == 3:
      splits = torch.utils.data.random_split(self.dataset['full'], split_dimension)
      self.dataset['train'], self.dataset['val'], self.dataset['test'] = splits[0], splits[1], splits[2]

    num_workers = os.cpu_count()

    #Assign batchsize
    self.train_batchsize = train_batchsize
    self.eval_batchsize = eval_batchsize

    
    self.dataloader = {}
    #Create the full dataloader of the dataset (it might be useful for experiments)
    self.dataloader['full'] = torch.utils.data.DataLoader(self.dataset['full'],
                                            batch_size = self.train_batchsize,
                                            shuffle = False, num_workers = num_workers)

    #If we ask for a split, the class will create train and validation dataloaders
    if self.n_splits > 1:
      self.dataloader['train'] = torch.utils.data.DataLoader(self.dataset['train'], 
                                                batch_size = self.train_batchsize, 
                                                shuffle = True, pin_memory = True, 
                                                num_workers = num_workers)
      
      self.dataloader['val'] = torch.utils.data.DataLoader(self.dataset['val'],
                                                batch_size = self.eval_batchsize,
                                                shuffle = False, num_workers = num_workers)
    
    #If we ask for two split, the class will create train, validation and test dataloaders
    if self.n_splits > 2:
      self.dataloader['test'] = torch.utils.data.DataLoader(self.dataset['test'],
                                            batch_size = self.eval_batchsize,
                                            shuffle = False, num_workers = num_workers)