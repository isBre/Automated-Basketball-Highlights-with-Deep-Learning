import torch.nn as nn
from torch import flatten
import torch.nn.functional as F

class BaselineModel(nn.Module):
    """
    A simple CNN that contains 3 convolutional layer followed respectively by
    1 maxpooling layer, then a global max pooling and at the end we have two linear layer.
    The output has dimension one, we have a single output that represent the probability
    to have a point given an image (I'll use a sigmoid at the end). 
    """

    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv2 = nn.Conv2d(6, 12, 3)
        self.conv3 = nn.Conv2d(12, 24, 3)

        self.max_pool = nn.MaxPool2d(3)

        self.adp_max_pool = nn.AdaptiveMaxPool2d(1)

        self.sigmoid = nn.Sigmoid()

        self.fc1 = nn.Linear(24, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = self.adp_max_pool(x)
        x = flatten(x, start_dim = 1)
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x