import argparse
from mmdet.apis import init_detector, inference_detector
from mmdet.core import DatasetEnum
import mmcv
import os
from tqdm import tqdm
import cv2
from pathlib import Path

config_file = 'projects/configs/co_dino/co_dino_5scale_swin_large_16e_o365tococo.py'

class_mapping = {
                'motorcycle': 0, 
                'car': 1,       
                'bus': 2,       
                'truck': 3      
            }

def save_predictions(image_name, results, output_file, width_image, height_image, class_names):
    # save format

    with open(output_file, 'a') as f:
        for class_ids, bboxes in enumerate(results):
            # change to center, width, height
            name_class = class_names[class_ids]
            if name_class in class_mapping:
                for box in bboxes:
                    if len(box) == 5:
                        x1, y1, x2, y2, score = box
                        if score >= 0.08:
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

def extract_data(input_dir, checkpoint_file):
    # Thiết lập đường dẫn
    filename_save = str(checkpoint_file).split("/")[-2]
    if 'nafnet' in input_dir:
        output_file = f"{args.output_dir}/" + filename_save + "_" + checkpoint_file.stem + "_nafnet.txt"
    else:
        output_file = f"{args.output_dir}/" + filename_save + "_" + checkpoint_file.stem + ".txt"
    model = init_detector(config_file, str(checkpoint_file), DatasetEnum.COCO, device='cuda:0')
    class_names = model.CLASSES
    # Xóa file predict.txt nếu đã tồn tại
    if os.path.exists(output_file):
        os.remove(output_file)
    
    # Xử lý từng ảnh trong thư mục
    for image_name in tqdm(os.listdir(input_dir)):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, image_name)
            image = cv2.imread(image_path)
            height_image, width_image, _ = image.shape
            results = inference_detector(model, image_path)
            save_predictions(image_name, results, output_file, width_image, height_image, class_names)
    print(checkpoint_file.name)
    print(input_dir)
    print(output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference and save predictions.")
    parser.add_argument("--path_private_test", type=str, required=True, help="Path to the input directory containing images.")
    parser.add_argument("--path_private_test_nafgs", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save the output files.")
    args = parser.parse_args()
    extract_data(input_dir=args.path_private_test_nafgs, checkpoint_file=Path("weight/weight_nafgs_withDat_exclude/epoch_3.pth"))
    extract_data(input_dir=args.path_private_test_nafgs, checkpoint_file=Path("weight/weight_nafnet_gsad_8ep/epoch_3.pth"))
    extract_data(input_dir=args.path_private_test_nafgs, checkpoint_file=Path("weight/weight_nafnet_gsad/epoch_15.pth"))
    extract_data(input_dir=args.path_private_test_nafgs, checkpoint_file=Path("weight/nafnet_gsad_pseudo_exclude/epoch_2.pth"))
    extract_data(input_dir=args.path_private_test, checkpoint_file=Path("weight/weight_psudo/epoch_4.pth"))
    extract_data(input_dir=args.path_private_test, checkpoint_file=Path("weight/weight_base/epoch_15.pth"))
    extract_data(input_dir=args.path_private_test, checkpoint_file=Path("weight/pseudo_version5_label/epoch_4.pth"))
    

