from ultralytics import YOLO

# Load a model
model = YOLO("yolov8x-seg.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(
    data="/home/timssh/ML/TAGGING/CLS/instance/Dataset-3/data.yaml",
    epochs=300,
    imgsz=(640, 480),
)
