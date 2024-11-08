import cv2
import os
import sys

# Adjust sys.path to include the project root directory
project_root = "/home/comflow-pc/Desktop/MVPCD"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

config = "config/config.yaml"

dataset_dir = 'data'
train_dir = os.path.join(project_root, config['output']['train_image_dir'])
val_dir = os.path.join(project_root, config['output']['val_image_dir'])
images_dir = os.path.join(dataset_dir, 'images', 'val')
labels_dir = os.path.join(dataset_dir, 'labels', 'val')
label_train_dir = os.path.join(project_root, config['output']['train_val_dir'])
label_val_dir = os.path.join(project_root, config['output']['val_val_dir'])

class_names = ['smallthing', 'woodthing', 'whitething', 'blackthing']

dirs = [train_dir, val_dir, label_train_dir, label_val_dir]

for dir in dirs:
    for filename in os.listdir(dir):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(dir, filename)
            label_path = os.path.join(labels_dir, f"{os.path.splitext(filename)[0]}.txt")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            print(f"Incorrect format in {label_path}: {line}")
                            continue
                        class_id, x_center, y_center, width, height = map(float, parts)
                        
                        img_h, img_w = image.shape[:2]
                        x_center *= img_w
                        y_center *= img_h
                        width *= img_w
                        height *= img_h
                        
                        x1 = int(x_center - width / 2)
                        y1 = int(y_center - height / 2)
                        x2 = int(x_center + width / 2)
                        y2 = int(y_center + height / 2)
                        
                        color = (0, 255, 0)  # Green bounding box
                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(image, class_names[int(class_id)], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            cv2.imshow('Image with Labels', image)
            cv2.waitKey(0)  # Press any key to move to the next image

cv2.destroyAllWindows()