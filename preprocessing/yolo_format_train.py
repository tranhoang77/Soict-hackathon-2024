import os
import shutil
import random
from tqdm import tqdm
import argparse

def adjust_to_divisible_by_four(number):
    """Adjust number to nearest value divisible by 4"""
    remainder = int(number) % 4
    return remainder

def process_label_file(file_path):
    """Process a single label file to adjust class numbers"""
    modified = False
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            if line.strip():
                parts = line.strip().split()
                if parts:
                    try:
                        current_number = int(parts[0])
                        adjusted_number = adjust_to_divisible_by_four(current_number)
                        
                        if current_number != adjusted_number:
                            modified = True
                            parts[0] = str(adjusted_number)
                        
                        new_lines.append(' '.join(parts) + '\n')
                    except ValueError:
                        new_lines.append(line)
            else:
                new_lines.append(line)
        
        if modified:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
            return True
        return False
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

def create_yolo_folder_structure(source_dir, source_dir_label, output_dir, delete_cam05):
    """Create YOLO dataset structure with class number adjustment"""
    try:
        # Create main directories
        dataset_dir = output_dir
        dirs = {
            "images/train": os.path.join(dataset_dir, "images", "train"),
            "images/val": os.path.join(dataset_dir, "images", "val"),
            "labels/train": os.path.join(dataset_dir, "labels", "train"),
            "labels/val": os.path.join(dataset_dir, "labels", "val")
        }
        
        # Create directories if they don't exist
        print("Creating directory structure...")
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Get all image files
        if delete_cam05:
            image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg') and "cam_05" not in f]
        else:
            image_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
        if not image_files:
            raise Exception("No jpg files found in source directory")
        
        random.shuffle(image_files)  # Shuffle files for random split
        
        # Calculate split indices (80% train, 20% val)
        split_idx = int(len(image_files) * 0.8)
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        # Function to copy and process files
        def copy_and_process_files(files, split_type):
            processed_count = 0
            with tqdm(total=len(files), desc=f"Processing {split_type} files") as pbar:
                for img_file in files:
                    try:
                        # Get corresponding label file
                        label_file = img_file.replace('.jpg', '.txt')
                        
                        # Source paths
                        img_src = os.path.join(source_dir, img_file)
                        label_src = os.path.join(source_dir_label, label_file)
                        
                        # Check if source files exist
                        if not os.path.exists(img_src) or not os.path.exists(label_src):
                            print(f"Warning: Missing files for {img_file}")
                            continue
                        
                        # Destination paths
                        img_dst = os.path.join(dirs[f"images/{split_type}"], img_file)
                        label_dst = os.path.join(dirs[f"labels/{split_type}"], label_file)
                        
                        # Copy files
                        shutil.copy2(img_src, img_dst)
                        shutil.copy2(label_src, label_dst)
                        
                        # Process label file
                        if process_label_file(label_dst):
                            processed_count += 1
                        
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error processing {img_file}: {str(e)}")
            
            return processed_count
        
        # Process files
        print("\nProcessing and copying files...")
        train_processed = copy_and_process_files(train_files, "train")
        val_processed = copy_and_process_files(val_files, "val")
        
        # Create data.yaml file
        print("\nCreating data.yaml...")
        yaml_content = f"""train: {os.path.join(dataset_dir, 'images', 'train')}
val: {os.path.join(dataset_dir, 'images', 'val')}
nc: 4  # number of classes (modify as needed)
names:   # class names (modify as needed)
    0: bike
    1: car
    2: coach
    3: container """
    
        with open(os.path.join(dataset_dir, 'data.yaml'), 'w') as f:
            f.write(yaml_content)
        
        # Print summary
        print("\nSummary:")
        print(f"Training images: {len(train_files)} (processed: {train_processed})")
        print(f"Validation images: {len(val_files)} (processed: {val_processed})")
        print(f"Dataset structure created in '{dataset_dir}' directory")
        print("\nProcess completed successfully!")

    except Exception as e:
        print(f"\nError: {str(e)}")
        print("Process failed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create YOLO dataset structure.")
    parser.add_argument("--train_dir_image", type=str, required=True, help="Path to training image directory")
    parser.add_argument("--train_dir_label", type=str, required=True, help="Path to training label directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to output directory for YOLO folder")
    parser.add_argument("--delete_cam05", type=bool, default=False, help="Path to output directory for YOLO folder")
    
    args = parser.parse_args()
    output_dir = os.path.join(args.output_dir, 'data')
    os.makedirs(output_dir, exist_ok=True)
    create_yolo_folder_structure(args.train_dir_image, args.train_dir_label, output_dir, args.delete_cam05)
