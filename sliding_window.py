"""
TEST SLIDING WINDOW METHOD
"""

import cv2
from torch import load, device, no_grad
from src.resnet import generate_resnet
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Initialize the model
model = generate_resnet(number=50)
weights_path = 'models/resnet50_cropped.pth'  # Path to your local .pth file
model.load_state_dict(load(weights_path, map_location=device('cpu')))
model.eval()

# Define data transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    transforms.Resize([128, 128]),
])

# Load the image
image = cv2.imread('datasets/Small-Frames/Point/vlcsnap-2022-10-30-15h58m14s468.png')

# Parameters
window_size = 128
stride_w = 20  # Horizontal stride
stride_h = 20  # Vertical stride

# List to store the patches' predictions and their coordinates
predictions = []
coordinates = []

# Loop over the image using sliding window
for y in range(0, image.shape[0] - window_size + 1, stride_h):
    for x in range(0, image.shape[1] - window_size + 1, stride_w):
        # Dynamically crop the patch
        patch = image[y:y + window_size, x:x + window_size]
        
        # Convert the patch to a tensor, add batch dimension, and preprocess it
        patch_tensor = data_transforms(patch).unsqueeze(0)  # Add batch dimension [1, 3, 128, 128]
        
        # Run the patch through the CNN model
        with no_grad():
            prediction = model(patch_tensor)
        
        # Extract the prediction value and add to the list
        predictions.append(prediction.item())
        coordinates.append((x, y))  # Store the coordinates of the patch

# Find the maximum prediction value and its index
max_prediction = max(predictions)
max_index = predictions.index(max_prediction)

# Get the corresponding coordinates of the patch with the highest prediction
max_coordinates = coordinates[max_index]

# Output the result
print(f"Max prediction value: {max_prediction}")
print(f"Patch with highest value starts at: {max_coordinates}")

# Draw a rectangle on the image to highlight the patch with the highest prediction
top_left_x, top_left_y = max_coordinates
bottom_right_x = top_left_x + window_size
bottom_right_y = top_left_y + window_size

# Draw a rectangle around the patch with the highest prediction value
cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)

# Convert the image to RGB format for displaying using matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with the highlighted patch
plt.figure(figsize=(10, 10))
plt.imshow(image_rgb)
plt.title(f"Max Prediction: {max_prediction} at ({top_left_x}, {top_left_y})")
plt.axis('off')  # Hide axis
plt.show()
