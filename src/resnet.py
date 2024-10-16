from torchvision.models import ResNet, resnet50, ResNet50_Weights
import torch.nn as nn

def generate_resnet50(
    pretrained=False,
    device="cpu",
) -> ResNet:
    '''
        Generate a ResNet50 model with a sigle parameter as output
        that is contained between [0, 1]

        Args:
        pretrained: if true use ResNet50_Weights.DEFAULT otherwise None
        device: str of the device used, if omitted is "cpu"
        return:
        resnet50 model
    '''


    if pretrained:
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model = resnet50(weights=None)
  
    #Implementation of the last layer
    #I need a sigmoid since I eventually want to represent a probability
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid()
    )
    
    model = model.to(device)
    return model