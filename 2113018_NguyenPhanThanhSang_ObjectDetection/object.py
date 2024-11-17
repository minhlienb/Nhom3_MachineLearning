# Import necessary libraries
import torch
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from colorama import Fore, Style
import torchvision

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
model_ = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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

# Function to draw bounding boxes
def draw_box(pred_class, img, rect_th=2, text_size=0.5, text_th=2, img_name="hinh2.png"):
    image = (np.clip(cv2.cvtColor(np.clip(img.numpy().transpose((1, 2, 0)), 0, 1), cv2.COLOR_RGB2BGR), 0, 1) * 255).astype(np.uint8).copy()
    
    for predicted_class in pred_class:
        label, probability, box = predicted_class
        t, l = round(box[0][0]), round(box[0][1])
        r, b = round(box[1][0]), round(box[1][1])
        
        print(f"\nLabel: {Fore.GREEN}{label}{Style.RESET_ALL}")
        print(f"Box coordinates: {t}, {l}, {r}, {b}")
        print(f"Probability: {probability}")
        
        # Draw rectangle and label
        cv2.rectangle(image, (t, l), (r, b), (0, 255, 0), rect_th)
        cv2.rectangle(image, (t, l), (t + 110, l + 17), (255, 255, 255), -1)
        cv2.putText(image, label + ": " + str(round(probability, 2)), (t + 10, l + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    
    # Save the resulting image
    cv2.imwrite(img_name, image)
    print(f"Result saved as {img_name}")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

# Load the image and transform it
image_path = 'Picture//0a9a02008bfcc32b.jpg'  # Replace with your image filename
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([transforms.ToTensor()])
img = transform(image)

# Perform predictions
pred = model_([img])
pred_class = get_predictions(pred, objects=COCO_INSTANCE_CATEGORY_NAMES)  

# Draw boxes
draw_box(pred_class, img)

# Save memory
del img, pred_class, pred
torch.cuda.empty_cache()
