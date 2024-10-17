# ~/Desktop/MVPCD/scripts/view_depth.py
import cv2
import numpy as np
import os
import yaml

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def view_depth_images(config):
    depth_dir = os.path.abspath(config['output']['depth_dir'])

    if not os.path.exists(depth_dir):
        print(f"Tiefenkarten-Verzeichnis existiert nicht: {depth_dir}")
        return

    for filename in sorted(os.listdir(depth_dir)):
        if filename.endswith('.npy'):
            depth_path = os.path.join(depth_dir, filename)
            depth_map = np.load(depth_path)

            # Anwenden der Tiefen-Cutoff-Werte
            min_depth = config.get('depth_threshold', {}).get('min', 500)  # Standardwert 500 mm
            max_depth = config.get('depth_threshold', {}).get('max', 2000)  # Standardwert 2000 mm
            depth_mask = np.zeros_like(depth_map, dtype=np.uint8)
            depth_mask[(depth_map >= min_depth) & (depth_map <= max_depth)] = 255

            # Optional: Nur die Bereiche innerhalb des Cutoffs anzeigen
            filtered_depth = cv2.bitwise_and(depth_map, depth_map, mask=depth_mask)

            # Normalisieren für die Anzeige
            depth_display = cv2.normalize(filtered_depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_display = depth_display.astype(np.uint8)
            depth_display = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

            cv2.imshow(f"Tiefenkarte: {filename}", depth_display)
            print(f"Zeige Tiefenkarte: {depth_path}")

            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()

            if key == ord('q'):
                print("Tiefenkartenanzeige beendet.")
                break

    cv2.destroyAllWindows()
    print("Alle Tiefenkarten wurden angezeigt.")

if __name__ == "__main__":
    config = load_config()
    view_depth_images(config)
