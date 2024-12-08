# object_Detection
#Implementation of SSD on custom data set using pretrained model 

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
import cv2
from xml.etree import ElementTree as ET
import torch
import torchvision
print(f"PyTorch version: {torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")


import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("PyTorch is using GPU:", torch.cuda.get_device_name(0))
else:
    print("PyTorch is using CPU.")


class PascalVOCDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.files = [f for f in os.listdir(root) if f.endswith('.jpg') or f.endswith('.png')]

        # Filter out images with no annotations
        self.files = [file for file in self.files if self._has_annotations(os.path.join(self.root, os.path.splitext(file)[0] + ".xml"))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_file = self.files[idx]
        img_path = os.path.join(self.root, img_file)
        annotation_path = os.path.splitext(img_path)[0] + ".xml"

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        boxes, labels = self._parse_xml(annotation_path)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels}
        img = F.to_tensor(img)

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def _parse_xml(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            boxes.append([xmin, ymin, xmax, ymax])

            label = obj.find("name").text
            labels.append(self._label_to_int(label))

        return boxes, labels

    def _label_to_int(self, label):
        label_map = {
            "name_of_class": 1    # enter the no of class that you have along with no 
        }                
        return label_map[label]

    def _has_annotations(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        # Check if there are any objects in the annotation
        return len(root.findall("object")) > 0

def get_ssd_model(num_classes):
    # Load pre-trained SSD model (MobileNetV3 backbone)
    model = ssdlite320_mobilenet_v3_large(pretrained=True)

    # Modify the head to match the number of classes
    num_classes = 2  # edit the no of class as per your data set 
    model.head.classification_head.num_classes = num_classes

    return model

# Paths to your dataset
train_data_path = r"C:\Users\saads\Desktop\saad_frcn\train"  # Update this path as per your computer directory 
val_data_path = r"C:\Users\saads\Desktop\saad_frcn\val"      #  Update this path as per your computer directory 

# Dataset and DataLoader
train_dataset = PascalVOCDataset(train_data_path)
val_dataset = PascalVOCDataset(val_data_path)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8  # 7 classes + 1 background  # edit the no of class as per your data set 
model = get_ssd_model(num_classes).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 10 # you can choose the no of epoch as per your requirement 
for epoch in range(num_epochs):
    model.train()
    for images, targets in train_loader:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {losses.item():.4f}")

# Save the model
torch.save(model.state_dict(), "ssd_model.pth")

# Load the model for inference
model.load_state_dict(torch.load("ssd_model.pth"))
model.eval()

import matplotlib.pyplot as plt

def predict_on_image(model, image_path, device, label_map, threshold=0.5):
    """
    Make predictions on a single image and display the results.
    
    Args:
        model: Trained SSD model.
        image_path: Path to the test image.
        device: Device (CPU or GPU) on which the model is running.
        label_map: Dictionary mapping class IDs to class names.
        threshold: Confidence threshold for displaying predictions.
    """
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    # Make predictions
    model.eval()
    with torch.no_grad():
        predictions = model(img_tensor)

    # Draw bounding boxes and labels
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > threshold:
            x1, y1, x2, y2 = box.int().tolist()
            class_name = label_map[label.item()]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} ({score:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

# Define the label map
label_map = {
   1: "fish"  # edit the label as per your class 
}

# Test the model on a single image
test_image_path = r"C:\Users\saads\Desktop\saad_frcn\train\IMG_2275_jpeg_jpg.rf.66355520a49ba7fb7082052f7ca6fee0.jpg"  # Update with your test image path
predict_on_image(model, test_image_path, device, label_map)

def predict_on_video(model, video_path, output_path, device):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        img_tensor = F.to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            predictions = model(img_tensor)

        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            if score > 0.5:  # Confidence threshold
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Class {label.item()}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        out.write(frame)

    cap.release()
    out.release()

# Example usage
video_path = r"C:\Users\saads\Desktop\saad_frcn\1.mp4"
output_path = r"C:\Users\saads\Desktop\saad_frcn\output1.mp4"
predict_on_video(model, video_path, output_path, device)

