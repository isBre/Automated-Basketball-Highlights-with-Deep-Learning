import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from torch import device


def bb_center(result):
    try:
        #TODO very ugly
        center_x = int(result.xyxy[0][0][0].item() + (result.xyxy[0][0][2].item() - result.xyxy[0][0][0].item())/2)
        center_y = int(result.xyxy[0][0][1].item() + (result.xyxy[0][0][3].item() - result.xyxy[0][0][1].item())/2)
        return center_x, center_y
    except:
        return None, None


def basket_found(result):
    return not (len(result.pred[0]) == 0)


def yolov5resnet50_pipeline(
      frame: np.ndarray,
      transform: transforms,
      object_detection_model,
      classifier_model,
      debug: bool = False,
    ) -> int:
    """
    IMPORTANT This function takes as input an image of type np.ndarray
    (so it works with frames extracted with OpenCV)
    """
    #Yolo was trainend on BGR images 
    y_result = object_detection_model(frame, size = 640)

    if debug:
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        print(y_result)

    #if there is a basket in the image, I can go on 
    #otherwise i can return confidence equal to zero
    if basket_found(y_result):

        #Obtain the center of the bounding box
        x, y = bb_center(y_result)

        #I need to convert image in RGB
        converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #Cropping the frame
        crop_frame = converted_frame[y-64:y+64, x-64:x+64, :]
        #I need to transform the input in tensor and normalize
        tensor_cropped_frame = transform(crop_frame).to(device)
        #I need to add an additional dimension to fit the model (is not in batch)
        input_img = tensor_cropped_frame[None, :]
        
        if debug:
            print(input_img.shape)
            print(transform(crop_frame))

        #Do the inference
        r_result = classifier_model(input_img)

        if debug:
            print(r_result)

        return r_result.item()

    #Else return 0
    return 0