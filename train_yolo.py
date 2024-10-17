from ultralytics import YOLO

# Load the pre-trained YOLO model
model = YOLO("yolo11n.pt")

# Train the model with the specified dataset and parameters
train_results = model.train(
    data="datasets/BasketLocalization/dataset.yaml",  # Path to the dataset YAML configuration
    epochs=50,  # Number of training epochs
    imgsz=640,  # Training image size (pixels)
    device="cpu",  # Specify the device for training (CPU in this case)
    patience=5,  # Early stopping after 5 epochs without improvement
)

# Export the trained model to ONNX format for deployment or further use
onnx_path = model.export(format="onnx")  # Returns the path to the exported ONNX model
