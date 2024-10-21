import os
import cv2

def view_annotations(image_dir, label_dir, class_names=None):
    # Get a list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image {image_path}")
            continue

        height, width = image.shape[:2]

        # Check if annotation file exists
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                annotations = f.readlines()

            for annotation in annotations:
                annotation = annotation.strip()
                if not annotation:
                    continue
                parts = annotation.split()
                if len(parts) != 5:
                    print(f"Invalid annotation in {label_path}: {annotation}")
                    continue

                # Parse annotation
                class_id, x_center, y_center, bbox_width, bbox_height = map(float, parts)

                # Convert normalized coordinates to pixel values
                x_center *= width
                y_center *= height
                bbox_width *= width
                bbox_height *= height

                # Calculate bounding box coordinates
                x_min = int(x_center - bbox_width / 2)
                y_min = int(y_center - bbox_height / 2)
                x_max = int(x_center + bbox_width / 2)
                y_max = int(y_center + bbox_height / 2)

                # Draw the bounding box
                color = (0, 255, 0)  # Green color for the bounding box
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

                # Put class label above the bounding box
                if class_names and int(class_id) < len(class_names):
                    label = class_names[int(class_id)]
                else:
                    label = str(int(class_id))
                cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                print("displaying " + image_path + " and " + label_path)
        else:
            print(f"No annotation file found for {image_path}")

        # Display the image with bounding boxes
        cv2.imshow('Annotated Image', image)
        key = cv2.waitKey(0)  # Wait for a key press to show the next image
        if key == 27:  # Press 'Esc' key to exit
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set the paths to your images and labels directories
    image_directory = 'MVPCD/data/images/train'
    label_directory = 'MVPCD/data/labels/train'

    # Optionally, provide a list of class names corresponding to class IDs
    class_names = ['class0', 'class1', 'class2']  # Replace with your actual class names

    view_annotations(image_directory, label_directory, class_names)
