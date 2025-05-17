import os
import cv2
import numpy as np
import yaml
import shutil
import re
from sklearn.model_selection import train_test_split
# Assuming src.logger and src.custom_exception are in your project's src directory
# If not, you might need to adjust these imports or remove the logging/exception handling
from src.logger import get_logger 
from src.custom_exception import CustomException

logger = get_logger(__name__)

def natural_sort_key(s):
    """Key function for natural sorting."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('(\\d+)', s)]

def prepare_yolo_dataset(original_root: str, yolo_base_dir: str, train_ratio: float = 0.8, class_names: list = None):
    """
    Prepares dataset for YOLO training.

    Args:
        original_root (str): Path to the root directory of the original dataset 
                             (containing "Images" and "Labels" subfolders).
        yolo_base_dir (str): Path to the directory where the YOLO formatted dataset will be stored.
        train_ratio (float): Ratio of training data to the whole dataset.
        class_names (list): List of class names. Defaults to ['gun'] if None.

    Returns:
        str: Path to the generated data.yaml file.
    """
    if class_names is None:
        class_names = ['gun']
    nc = len(class_names)

    logger.info(f"Starting YOLO dataset preparation. Original root: {original_root}, YOLO base: {yolo_base_dir}")

    original_image_dir = os.path.join(original_root, "Images")
    original_label_dir = os.path.join(original_root, "Labels")

    if not os.path.isdir(original_image_dir):
        raise CustomException(f"Original image directory not found: {original_image_dir}", None)
    if not os.path.isdir(original_label_dir):
        raise CustomException(f"Original label directory not found: {original_label_dir}", None)

    # Create YOLO directory structure
    images_dir = os.path.join(yolo_base_dir, "images")
    labels_dir = os.path.join(yolo_base_dir, "labels")

    for sub_dir in ("train", "val"):
        os.makedirs(os.path.join(images_dir, sub_dir), exist_ok=True)
        os.makedirs(os.path.join(labels_dir, sub_dir), exist_ok=True)

    # Pair images and labels
    image_filenames = sorted(os.listdir(original_image_dir), key=natural_sort_key)
    file_pairs = []
    for img_name in image_filenames:
        base, ext = os.path.splitext(img_name)

        label_name = base + ".txt" 
        
        if os.path.exists(os.path.join(original_label_dir, label_name)):
            file_pairs.append((img_name, label_name))
        else:
            logger.warning(f"Label file not found for image: {img_name}. Skipping this image.")

    if not file_pairs:
        raise CustomException("No image-label pairs found. Check image and label names and paths.", None)

    # Split files
    train_files, val_files = train_test_split(file_pairs, train_size=train_ratio, random_state=42, shuffle=True)
    logger.info(f"Found {len(file_pairs)} total image-label pairs. Split into {len(train_files)} train and {len(val_files)} validation files.")

    def process_files(file_list, image_dest_subdir, label_dest_subdir):
        for img_filename, label_filename in file_list:
            # Copy image
            src_img_path = os.path.join(original_image_dir, img_filename)
            dst_img_path = os.path.join(images_dir, image_dest_subdir, img_filename)
            shutil.copy(src_img_path, dst_img_path)

            # Read original image dimensions
            img = cv2.imread(src_img_path)
            if img is None:
                logger.error(f"Could not read image: {src_img_path}. Skipping.")
                continue
            img_height, img_width = img.shape[:2]

            # Process label file
            src_label_path = os.path.join(original_label_dir, label_filename)
            yolo_labels = []
            try:
                with open(src_label_path, "r") as f:
                    lines = f.read().strip().splitlines()
                
                if not lines:
                    logger.warning(f"Label file {label_filename} is empty. Creating empty label file for YOLO.")
                
                # First line is often count, but can be malformed. Be robust.
                try:
                    num_annotations = int(lines[0])
                    annotations = lines[1:num_annotations+1] # Process only declared annotations
                except (ValueError, IndexError): # If first line is not a count or empty
                    logger.warning(f"First line of {label_filename} is not a valid count or file is structured differently. Attempting to parse all lines as bounding boxes.")
                    annotations = lines # Assume all lines (or remaining lines) are annotations

                for line in annotations:
                    parts = list(map(int, line.split()))
                    if len(parts) == 4: # x1, y1, x2, y2
                        x1, y1, x2, y2 = parts
                        
                        # Convert to YOLO format: class_id cx cy w h (normalized)
                        # Assuming class_id is 0 for 'gun' as per nc=1
                        class_id = 0 
                        
                        box_width = x2 - x1
                        box_height = y2 - y1
                        center_x = x1 + box_width / 2
                        center_y = y1 + box_height / 2

                        # Normalize
                        norm_center_x = center_x / img_width
                        norm_center_y = center_y / img_height
                        norm_width = box_width / img_width
                        norm_height = box_height / img_height
                        
                        yolo_labels.append(f"{class_id} {norm_center_x:.6f} {norm_center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
                    else:
                        logger.warning(f"Skipping malformed line in {label_filename}: '{line}'")

            except Exception as e:
                logger.error(f"Error processing label file {src_label_path}: {e}")
                # Create an empty label file if processing fails to avoid missing label errors in YOLO
                # but log the error.
            
            # Write YOLO label file (even if empty)
            # Ensure the label filename matches the image filename (except extension)
            base_img_filename, _ = os.path.splitext(img_filename)
            dst_label_path = os.path.join(labels_dir, label_dest_subdir, base_img_filename + ".txt")
            with open(dst_label_path, "w") as f:
                f.write("\n".join(yolo_labels))

    logger.info("Processing training files...")
    process_files(train_files, "train", "train")
    logger.info("Processing validation files...")
    process_files(val_files, "val", "val")

    # Create data.yaml
    yaml_config = {
        'path': os.path.abspath(yolo_base_dir),  # Absolute path to YOLO dataset root
        'train': 'images/train',  # Relative path from 'path'
        'val': 'images/val',      # Relative path from 'path'
        'nc': nc,
        'names': class_names
    }

    yaml_path = os.path.join(yolo_base_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_config, f, sort_keys=False, default_flow_style=False)
    
    logger.info(f"YOLO dataset preparation complete. data.yaml created at: {yaml_path}")
    return yaml_path

if __name__ == "__main__":
    
    try:
        # These paths need to exist on your system
        original_data_root = "artifacts/raw" # Contains Images/ and Labels/
        yolo_dataset_dir = "artifacts/gun_dataset_yolo" # Will be created

        if not os.path.exists(os.path.join(original_data_root, "Images")) or \
           not os.path.exists(os.path.join(original_data_root, "Labels")):
            print(f"Error: Ensure '{original_data_root}/Images' and '{original_data_root}/Labels' exist.")
            # Create dummy files for testing if they don't exist
            print("Creating dummy data for testing purposes...")
            os.makedirs(os.path.join(original_data_root, "Images"), exist_ok=True)
            os.makedirs(os.path.join(original_data_root, "Labels"), exist_ok=True)
            # Create a dummy image
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(original_data_root, "Images", "gun_1.jpeg"), dummy_image)
            # Create a dummy label file
            with open(os.path.join(original_data_root, "Labels", "gun_1.txt"), "w") as f:
                f.write("1\n10 10 50 50") # 1 annotation, x1 y1 x2 y2
            print(f"Dummy image: {os.path.join(original_data_root, 'Images', 'gun_1.jpeg')}")
            print(f"Dummy label: {os.path.join(original_data_root, 'Labels', 'gun_1.txt')}")


        data_yaml_path = prepare_yolo_dataset(
            original_root=original_data_root,
            yolo_base_dir=yolo_dataset_dir,
            train_ratio=0.8,
            class_names=['gun']
        )
        print(f"Dataset preparation successful. YOLO config: {data_yaml_path}")
    except CustomException as e:
        print(f"A custom error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
