import cv2
from tqdm import tqdm
from scipy import signal
from ultralytics import YOLO
from inference import pipeline
from torchvision import transforms
from torch.cuda import is_available
from src.resnet import generate_resnet
from torch import device, load, no_grad

VIDEO_PATH = 'datasets/altamura.mp4'
VIDEO_FPS = 30

# Device selection: use GPU if available
CURRENT_DEVICE = device("cuda:0" if is_available() else "cpu")
print(f"Device Selected: {CURRENT_DEVICE}")

RESNET_WEIGHT = 'models/resnet50_cropped.pth'
YOLO_WEIGHT = "models/yolov11_best.pt"

resnet50 = generate_resnet(number=50, current_device=CURRENT_DEVICE)
resnet50.load_state_dict(load(RESNET_WEIGHT, map_location=CURRENT_DEVICE))
resnet50.eval()

yolo11 = YOLO(YOLO_WEIGHT, task='detect').to(CURRENT_DEVICE)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    )])

frame_confidence = []

cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = 2000 #int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with no_grad():
    for i in tqdm(range(total_frames)):

        ret, frame = cap.read()
        
        # This try is useful since sometimes the shape of the bounding box is too small
        try:
            result = pipeline(
                frame=frame, 
                transform=transform, 
                object_detection_model=yolo11, 
                classifier_model=resnet50,
                current_device=CURRENT_DEVICE,
            )
            frame_confidence.append(result)

        # Calculated padded input size per channel: (6 x 134). Kernel size: (7 x 7). 
        # Kernel size can't be greater than actual input size
        except RuntimeError:
            frame_confidence.append(0.0)
    
cap.release()

print(frame_confidence)

frame_tollerance = VIDEO_FPS * 3
peak_indices = signal.find_peaks(frame_confidence, height=.5, distance = frame_tollerance)
peaks_list = peak_indices[0]

print(f"Total Peaks Found: {len(peaks_list)}")

