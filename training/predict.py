import argparse
from mmdet.apis import init_detector, inference_detector
from mmdet.core import DatasetEnum
import mmcv
import os
from tqdm import tqdm
import cv2

def save_predictions(image_name, results, output_file, width_image, height_image, class_names):
    class_mapping = {
        'motorcycle': 0,
        'car': 1,       
        'bus': 2,       
        'truck': 3       
    }

    with open(output_file, 'a') as f:
        for class_ids, bboxes in enumerate(results):
            try: 
                name_class = class_names[class_ids]
                if name_class in class_mapping:
                    for box in bboxes:
                        if len(box) == 5:
                            x1, y1, x2, y2, score = box
                            if score >= 0.1:
                                x_center = (box[0] + box[2]) / 2
                                y_center = (box[1] + box[3]) / 2
                                width = box[2] - box[0]
                                height = box[3] - box[1]
                
                                x_center /= width_image
                                y_center /= height_image
                                width /= width_image
                                height /= height_image
                
                                class_id = class_mapping[name_class]
                                line = f"{image_name} {class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
                                f.write(line)
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Inference script for object detection.")
    parser.add_argument('--config_file', type=str, default='projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py', help="Path to the config file (optional).")
    parser.add_argument('--checkpoint_file', type=str, required=True, help="Path to the checkpoint file.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the public test dir.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output predictions.")

    args = parser.parse_args()

    if args.config_file:
        model = init_detector(args.config_file, args.checkpoint_file, DatasetEnum.COCO, device='cuda:0')
    else:
        model = init_detector(None, args.checkpoint_file, DatasetEnum.COCO, device='cuda:0')

    class_names = model.CLASSES
    print(class_names)

    if os.path.exists(args.output_file):
        os.remove(args.output_file)
    
    for image_name in tqdm(os.listdir(args.input_dir)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_dir, image_name)
            image = cv2.imread(image_path)
            height_image, width_image, _ = image.shape
            results = inference_detector(model, image_path)
            save_predictions(image_name, results, args.output_file, width_image, height_image, class_names)

if __name__ == "__main__":
    main()
