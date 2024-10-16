from torchvision.models import (ResNet, ResNet50_Weights, ResNet18_Weights,
                                resnet50, resnet18)
import torch.nn as nn
from torch import device
from typing import Union

def generate_resnet(
        number = Union[18, 50],
        pretrained = False,
) -> ResNet:
    '''
        Generate a ResNet18 or ResNet50 model with a single parameter output
        that is contained between [0, 1].

        Args:
        number: int, specify either 18 for ResNet18 or 50 for ResNet50.
        pretrained: bool, if True, use pretrained weights (ResNet18_Weights.DEFAULT or ResNet50_Weights.DEFAULT), 
        otherwise use random initialization.
        
        Returns:
        A ResNet model with the final layer modified for a single output (a probability between 0 and 1).
    '''

    if number == 18:
        if pretrained:
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        else:
            model = resnet18(weights=None)
    elif number == 50:
        if pretrained:
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            model = resnet50(weights=None)
    else:
        raise ValueError("Invalid value for 'number'. Choose either 18 or 50.")
  
    #Implementation of the last layer
    #I need a sigmoid since I eventually want to represent a probability
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    model = model.to(device)
    return model