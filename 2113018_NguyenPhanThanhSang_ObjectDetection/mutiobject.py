# Import necessary libraries
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from colorama import Fore, Style
import torchvision
import os
import csv

# Define COCO categories
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load the model
model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="FasterRCNN_ResNet50_FPN_Weights.DEFAULT")
model_.eval()

# Function to get predictions
def get_predictions(pred, threshold=0.1, objects=None):
    predicted_classes = [(COCO_INSTANCE_CATEGORY_NAMES[i], p, [(box[0], box[1]), (box[2], box[3])])
                         for i, p, box in zip(pred[0]['labels'].numpy(), pred[0]['scores'].detach().numpy(),
                                              pred[0]['boxes'].detach().numpy())]
    predicted_classes = [stuff for stuff in predicted_classes if stuff[1] > threshold]
    if objects and predicted_classes:
        predicted_classes = [(name, p, box) for name, p, box in predicted_classes if name in objects]
    return predicted_classes

# # Function to draw bounding boxes
# def draw_box(pred_class, img, img_name="hinh.png"):
#     image = (np.clip(cv2.cvtColor(np.clip(img.numpy().transpose((1, 2, 0)), 0, 1), cv2.COLOR_RGB2BGR), 0, 1) * 255).astype(np.uint8).copy()
#     for predicted_class in pred_class:
#         label, probability, box = predicted_class
#         t, l = round(box[0][0]), round(box[0][1])
#         r, b = round(box[1][0]), round(box[1][1])

#         # Draw rectangle and label
#         cv2.rectangle(image, (t, l), (r, b), (0, 255, 0), 2)
#         cv2.rectangle(image, (t, l), (t + 110, l + 17), (255, 255, 255), -1)
#         cv2.putText(image, label + ": " + str(round(probability, 2)), (t + 10, l + 12),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=2)

#     # Save the resulting image
#     cv2.imwrite(img_name, image)
#     print(f"Result saved as {img_name}")

# Folder containing images
folder_path = 'D:/Các Phương Pháp Học Máy/Project/Picture'
output_csv = 'output.csv'

# Create a CSV file and write the header
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image Name", "Label", "Probability", "Top-Left (x, y)", "Bottom-Right (x, y)"])

    # Process each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        
        # Load and preprocess the image
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(image)

        # Perform predictions
        pred = model_([img])
        pred_class = get_predictions(pred, objects=COCO_INSTANCE_CATEGORY_NAMES)

        # Draw boxes and save the image with boxes
        # draw_box(pred_class, img, img_name=os.path.join(folder_path, f"boxed_{image_name}"))

        # Write data to CSV
        for item in pred_class:
            label, probability, box = item
            writer.writerow([image_name, label, round(probability, 2), box[0], box[1]])

        # Free up memory
        del img, pred_class, pred
        torch.cuda.empty_cache()

print(f"Results saved to {output_csv}")
