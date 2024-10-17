import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch import device, Tensor


def get_bb_center(xyxy: Tensor):
    """
    Calculate the center of the bounding box from the 'xyxy' tensor.
    
    Args:
    xyxy (torch.Tensor): Tensor of shape [1, 4] containing [x_min, y_min, x_max, y_max].
    
    Returns:
    torch.Tensor: A tensor containing the [x_center, y_center] coordinates of the bounding box center.
    """
    # Extract the x_min, y_min, x_max, y_max from the tensor
    x_min, y_min, x_max, y_max = xyxy[0]
    
    # Calculate the center of the bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    
    # Return the center as a tensor
    return int(x_center.item()), int(y_center.item())


def net_found(results):
    return results.boxes.cls.numel() != 0


def pipeline(
      frame: np.ndarray,
      transform: transforms,
      object_detection_model,
      classifier_model,
      current_device: device,
    ) -> float:
    """
    IMPORTANT This function takes as input an image of type np.ndarray
    (so it works with frames extracted with OpenCV)
    """
    # Yolo was trainend on BGR images 
    y_result = object_detection_model(frame, verbose=False)[0]
    
    # If there is a basket in the image, I can go on 
    # otherwise i can return confidence equal to zero
    if net_found(y_result):

        # Obtain the center of the bounding box
        x, y = get_bb_center(y_result.boxes.xyxy)

        # I need to convert image in RGB
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Cropping the frame
        crop_frame = converted_frame[y-64:y+64, x-64:x+64, :]
        # I need to transform the input in tensor and normalize
        tensor_cropped_frame = transform(crop_frame).to(current_device)
        # I need to add an additional dimension to fit the model (is not in batch)
        input_img = tensor_cropped_frame[None, :]

        # Do the inference
        r_result = classifier_model(input_img)
        return r_result.item()

    #Else return 0
    return 0.0