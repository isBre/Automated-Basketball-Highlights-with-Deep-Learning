from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import torch.nn as nn
from torch import sigmoid

# Define the model: MobileNetV3 for binary classification
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        # Load pre-trained MobileNetV3
        self.mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Modify the final layer for binary classification
        # The original classifier outputs many classes, we need 1 output for binary classification
        in_features = self.mobilenet.classifier[3].in_features  # Access the correct layer for MobileNetV3
        self.mobilenet.classifier[3] = nn.Linear(in_features, 1)  # Replace with 1 output for binary classification
    
    def forward(self, x):
        x = self.mobilenet(x)
        return sigmoid(x)  # Sigmoid to get output between 0 and 1