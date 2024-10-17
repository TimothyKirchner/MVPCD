# ~/Desktop/MVPCD/utils/dataset.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    boxes.append([x_center, y_center, width, height])
                    labels.append(int(class_id))
        else:
            print(f"Warning: Label file {label_name} does not exist for image {img_name}")

        # Convert boxes to Albumentations format (x_min, y_min, x_max, y_max)
        height, width = image.shape[:2]
        boxes_alb = []
        for box in boxes:
            x_center, y_center, w, h = box
            x_min = (x_center - w / 2) * width
            y_min = (y_center - h / 2) * height
            x_max = (x_center + w / 2) * width
            y_max = (y_center + h / 2) * height
            boxes_alb.append([x_min, y_min, x_max, y_max])

        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes_alb, class_labels=labels)
            image = transformed['image']
            boxes_alb = transformed['bboxes']
            labels = transformed['class_labels']

        # Convert boxes back to YOLO format
        boxes_yolo = []
        for box in boxes_alb:
            x_min, y_min, x_max, y_max = box
            x_center = ((x_min + x_max) / 2) / width
            y_center = ((y_min + y_max) / 2) / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height
            boxes_yolo.append([x_center, y_center, w, h])

        boxes = torch.tensor(boxes_yolo, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        return image, boxes, labels
