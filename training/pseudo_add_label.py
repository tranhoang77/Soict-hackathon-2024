import os
from mmdet.apis import init_detector, inference_detector
from mmdet.core import DatasetEnum
import cv2
from tqdm import tqdm
import argparse



def init_model():
    config_file = 'projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py'
    checkpoint_file = 'weight/weight_2e_for_pseudo/epoch_2.pth'
    model = init_detector(config_file, checkpoint_file, DatasetEnum.COCO, device='cuda:0')
    return model

def read_existing_annotations(txt_path):
    """Read existing annotations from txt file"""
    objects = set()
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id, x, y, w, h
                    # Convert strings to float for later calculations
                    objects.add(tuple(float(x) for x in parts))
    return objects

def calculate_iou(boxA, boxB):
    """Calculate IoU between two boxes"""
    x1_intersect = max(boxA[0], boxB[0])
    y1_intersect = max(boxA[1], boxB[1])
    x2_intersect = min(boxA[2], boxB[2])
    y2_intersect = min(boxA[3], boxB[3])

    # Calculate intersection area
    intersect_width = max(0, x2_intersect - x1_intersect)
    intersect_height = max(0, y2_intersect - y1_intersect)
    area_intersect = intersect_width * intersect_height

    area_boxA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    area_boxB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    area_union = area_boxA + area_boxB - area_intersect

    # Calculate IoU
    iou = area_intersect / area_union if area_union > 0 else 0
    return iou

def check_exist_bounding_box(boxA, existing_objects, class_id):
    """Check if a bounding box already exists in annotations"""
    class_id = int(class_id)
    for inf_object in existing_objects:
        id_object_in_txt = int(inf_object[0]) % 4
        # Convert center format to corner format
        x1 = inf_object[1] - inf_object[3] / 2  # center_x - width/2
        y1 = inf_object[2] - inf_object[4] / 2  # center_y - height/2
        x2 = inf_object[1] + inf_object[3] / 2  # center_x + width/2
        y2 = inf_object[2] + inf_object[4] / 2  # center_y + height/2
        boxB = (x1, y1, x2, y2)
        if calculate_iou(boxA, boxB) >= 0.3:
            return False
    return True

def process_detections(model, image_path, txt_path, txt_path_to_save, class_mapping):
    """Process detections and compare with existing annotations"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(txt_path_to_save), exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
        
    height_image, width_image, _ = image.shape
    
    # Get existing annotations
    existing_objects = read_existing_annotations(txt_path)

    with open(txt_path_to_save, 'w') as f:  # 'w' mode to start fresh
        for obj in existing_objects:
            # obj is a tuple of (class_id, x_center, y_center, width, height)
            line = f"{int(obj[0])} {obj[1]:.6f} {obj[2]:.6f} {obj[3]:.6f} {obj[4]:.6f}\n"
            f.write(line)

    # Get new detections
    results = inference_detector(model, image_path)
    detections = []

    # Collect all detections with calculated area
    for class_ids, bboxes in enumerate(results):
        name_class = model.CLASSES[class_ids]
        if name_class in class_mapping:
            class_id = class_mapping[name_class]
            
            for box in bboxes:
                if len(box) == 5:
                    x1, y1, x2, y2, score = box
                    
                    # Normalize coordinates
                    x1 = x1 / width_image
                    y1 = y1 / height_image
                    x2 = x2 / width_image
                    y2 = y2 / height_image

                    if score >= 0.2:
                        width = x2 - x1
                        height = y2 - y1
                        area = width * height
                        detections.append((class_id, x1, y1, x2, y2, area))

    # Sort detections by area in descending order
    detections.sort(key=lambda x: x[5], reverse=True)  # Sort by area

    # Process each sorted detection
    for det in detections:
        class_id, x1, y1, x2, y2, _ = det
        boxA = (x1, y1, x2, y2)  # Normalized coordinates
        
        if check_exist_bounding_box(boxA, existing_objects, class_id):
            # Convert to center format
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Add to existing_objects
            existing_objects.add((class_id, x_center, y_center, width, height))

            # Write to file
            with open(txt_path_to_save, 'a') as f:
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                f.write(line)


def main():
    # Paths
    parser = argparse.ArgumentParser(description="pseudo for object detection.")
    parser.add_argument('--image_dir', type=str, required=True, help="path to dir train image")
    parser.add_argument('--output_folder_label', type=str, required=True, help="Path to the save label new.")

    args = parser.parse_args()
    if not os.path.exists(args.output_folder_label):
        os.makedirs(args.output_folder_label, exist_ok=True)
    
    # Class mapping
    class_mapping = {
        'motorcycle': 0,
        'car': 1,
        'bus': 2,
        'truck': 3
    }
    
    # Initialize model
    model = init_model()
    
    # Process each image
    for image_name in tqdm(os.listdir(args.image_dir)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.image_dir, image_name)
            txt_name = os.path.splitext(image_name)[0] + '.txt'
            txt_path = os.path.join(args.image_dir, txt_name)  # Fixed typo in os.path
            txt_path_to_save = os.path.join(args.output_folder_label, txt_name)
            
            process_detections(model, image_path, txt_path, txt_path_to_save, class_mapping)

if __name__ == "__main__":
    main()