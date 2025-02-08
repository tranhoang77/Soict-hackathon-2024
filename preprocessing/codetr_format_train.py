import os
import json
from PIL import Image
import datetime
import random
import shutil
import argparse  # Import argparse

class YOLOtoCOCO:
    def __init__(self, train_dir_image, train_dir_label, output_dir, val_split=0.2, delete_cam05=False):
        """
        Initialize the converter
        :param train_dir: Directory containing both images and label files
        :param output_dir: Directory to save COCO format JSONs and split datasets
        :param val_split: Validation set ratio (default: 0.2 for 20%)
        :param seed: Random seed for reproducibility
        """
        self.train_dir_image = train_dir_image
        self.train_dir_label = train_dir_label
        self.output_dir = output_dir
        self.val_split = val_split
        self.delete_cam05 = delete_cam05
        
        # Create output directories
        self.train_output_dir = os.path.join(output_dir, 'train2017')
        self.val_output_dir = os.path.join(output_dir, 'val2017')
        self.annotation_dir = os.path.join(output_dir, 'annotations')
        os.makedirs(self.train_output_dir, exist_ok=True)
        os.makedirs(self.val_output_dir, exist_ok=True)
        os.makedirs(self.annotation_dir, exist_ok=True)

        # Initialize data structures
        self.train_images = []
        self.train_annotations = []
        self.val_images = []
        self.val_annotations = []
        self.categories = []
        self.category_mapping = {}
        
        # Define vehicle category mapping
        self.label_mapping = {
            '0': 'motorcycle',
            '1': 'car',
            '2': 'bus',
            '3': 'truck',
            '4': 'motorcycle',
            '5': 'car',
            '6': 'bus',
            '7': 'truck'
        }
        
        # Initialize categories with vehicle types
        unique_categories = []
        for label in self.label_mapping.values():
            if label not in unique_categories:
                unique_categories.append(label)

        for idx, category in enumerate(unique_categories):  
            self.category_mapping[category] = idx 
            self.categories.append({
                'id': idx, 
                'name': category,
                'supercategory': 'vehicle'
            })

    def get_image_info(self, image_path, image_id):
        """Get image information"""
        image = Image.open(image_path)
        width, height = image.size
        image_info = {
            'id': image_id,
            'file_name': os.path.basename(image_path),
            'width': width,
            'height': height,
            'date_captured': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'license': 1,
            'coco_url': "",
            'flickr_url': ""
        }
        return image_info

    def get_category_id(self, category_name):
        """Get category id from name"""
        return self.category_mapping[category_name]

    def convert_bbox_yolo_to_coco(self, bbox, img_width, img_height):
        """
        Convert YOLO bbox to COCO format
        YOLO: [x_center, y_center, width, height] (normalized)
        COCO: [x_min, y_min, width, height] (absolute)
        """
        x_center, y_center, width, height = bbox
        
        x_center = float(x_center) * img_width
        y_center = float(y_center) * img_height
        width = float(width) * img_width
        height = float(height) * img_height
        
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        
        return [x_min, y_min, width, height]

    def create_coco_annotations(self, image_files, is_train=True):
        """Process images and create COCO annotations"""
        annotations = []
        images = []
        annotation_id = 0
        
        for image_id, image_file in enumerate(image_files):
            base_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(self.train_dir_image, image_file)
            label_path = os.path.join(self.train_dir_label, f"{base_name}.txt")
            
            if not os.path.exists(label_path):
                print(f"Warning: No label file found for {image_file}")
                continue

            # Copy image to appropriate output directory
            dst_dir = self.train_output_dir if is_train else self.val_output_dir
            shutil.copy2(image_path, os.path.join(dst_dir, image_file))

            # Get image info
            image_info = self.get_image_info(image_path, image_id)
            images.append(image_info)

            # Process annotations
            with open(label_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                        
                    label_id = parts[0]
                    if label_id not in self.label_mapping:
                        print(f"Warning: Unknown label ID {label_id} in {image_file}")
                        continue
                        
                    category_name = self.label_mapping[label_id]
                    bbox = list(map(float, parts[1:5]))
                    
                    coco_bbox = self.convert_bbox_yolo_to_coco(
                        bbox, image_info['width'], image_info['height'])
                    
                    annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': self.get_category_id(category_name),
                        'bbox': coco_bbox,
                        'area': coco_bbox[2] * coco_bbox[3],
                        'segmentation': [],
                        'iscrowd': 0
                    }
                    
                    annotations.append(annotation)
                    annotation_id += 1
            
            print(f"Processed {image_file} for {'training' if is_train else 'validation'} set")
            
        return images, annotations

    def save_coco_format(self, images, annotations, output_path):
        """Save annotations in COCO format"""
        coco_format = {
            'info': {
                'year': 2024,
                'version': '1.0',
                'description': 'Vehicle detection dataset converted from YOLO format',
                'contributor': '',
                'url': '',
                'date_created': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'images': images,
            'annotations': annotations,
            'categories': self.categories
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_format, f, indent=2)

    def convert(self):
        """Convert YOLO annotations to COCO format with train/val split"""
        # Get all image files
        if self.delete_cam05:
            image_files = [f for f in os.listdir(self.train_dir_image) if f.lower().endswith(('.jpg', '.jpeg', '.png')) and "cam_05" not in f]
        else:
            image_files = [f for f in os.listdir(self.train_dir_image) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()

        # Split files into train and validation sets
        num_val = int(len(image_files) * self.val_split)
        random.shuffle(image_files)
        val_files = image_files[:num_val]
        train_files = image_files[num_val:]

        # Process training set
        print("\nProcessing training set...")
        self.train_images, self.train_annotations = self.create_coco_annotations(train_files, is_train=True)

        # Process validation set
        print("\nProcessing validation set...")
        self.val_images, self.val_annotations = self.create_coco_annotations(val_files, is_train=False)

        # Save annotations
        train_json_path = os.path.join(self.annotation_dir, 'instances_train2017.json')
        val_json_path = os.path.join(self.annotation_dir, 'instances_val2017.json')
        
        self.save_coco_format(self.train_images, self.train_annotations, train_json_path)
        self.save_coco_format(self.val_images, self.val_annotations, val_json_path)

        # Print summary
        print("\nConversion completed!")
        print(f"Training set: {len(self.train_images)} images, {len(self.train_annotations)} annotations")
        print(f"Validation set: {len(self.val_images)} images, {len(self.val_annotations)} annotations")
        print("\nCategory mapping:")
        for category in self.categories:
            print(f"- {category['name']} (ID: {category['id']})")
            count_train = sum(1 for ann in self.train_annotations if ann['category_id'] == category['id'])
            count_val = sum(1 for ann in self.val_annotations if ann['category_id'] == category['id'])
            print(f"  Train: {count_train}, Val: {count_val}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert YOLO annotations to COCO format")
    parser.add_argument("--train_dir_image", required=True, help="Path to the directory containing training images")
    parser.add_argument("--train_dir_label", required=True, help="Path to the directory containing YOLO label files")
    parser.add_argument("--output_dir", required=True, help="Output directory need do mount Co-DETR folder")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    parser.add_argument("--delete_cam05", type=bool, default=False, help="if you want delete cam 05")

    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, 'data/coco')
    os.makedirs(output_dir, exist_ok=True)

    # Initialize the converter with arguments from the command line
    converter = YOLOtoCOCO(
        train_dir_image=args.train_dir_image,
        train_dir_label=args.train_dir_label,
        output_dir=output_dir,
        val_split=args.val_split,
        delete_cam05=args.delete_cam05
    )
    converter.convert()