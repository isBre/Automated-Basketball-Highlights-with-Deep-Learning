import numpy as np
from typing import List
from torch import Tensor
import matplotlib.pyplot as plt
from utils import frame_to_hms

def imshow_tensor(tensor_image: Tensor) -> None:
    """
    Display a tensor image
    Args:
        tensor_image: tensor image
    """
    # Denormalization
    tensor_image = tensor_image / 2 + 0.5
    npimg = tensor_image.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def display_history(history : dict) -> None:
    """
    Plot the history of a given training

    Args:
        history: a dictiory that contain
        history['train_loss_values'] -> train loss history
        history['val_loss_values'] -> val loss history
        history['train_acc_values'] -> train accuracy history
        history['val_acc_values'] -> val accuracy history
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))

    # Visualize the behavior of the loss
    axes[0].plot(history['train_loss_values'])
    axes[0].plot(history['val_loss_values'])
    axes[0].set_title('Loss during training')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Training', 'Validation'])

    # and of the accuracy
    axes[1].plot(history['train_acc_values'])
    axes[1].plot(history['val_acc_values'])
    axes[1].set_title('Accuracy during training')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(['Training', 'Validation'])

    fig.tight_layout()
    plt.show()

def plot_frames_points(
        predicted_frames: List[int],
        true: List[int]=None,
        fps: int=60,
    ) -> None:

    """
    Plot the result of the yolov5+resnet50 model and the point frame count

    Args:
        predicted_frames: a list with len = N (number of total frames in a particular
                        video). Each element contain the probability of a point
        true: contain approximate frame in which we have a point (len << N),
            usually there are 40-70 points
    """
    if true is not None:
        for i in true:
            plt.axvline(x=i, color='r')
            plt.text(i+250, 0.87, str(f"{i} ({frame_to_hms(i, fps)})"), rotation=90)

    #Threshold line
    plt.axhline(y=0.5, color='green', linestyle='dotted', linewidth=0.5)

    plt.plot(list(range(len(predicted_frames))), predicted_frames)
    plt.show()