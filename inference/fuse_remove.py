import numpy as np
import argparse

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    bbox format: [x_center, y_center, width, height]
    """
    # Convert center format to corner format
    b1_x1, b1_y1 = bbox1[0] - bbox1[2]/2, bbox1[1] - bbox1[3]/2
    b1_x2, b1_y2 = bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
    
    b2_x1, b2_y1 = bbox2[0] - bbox2[2]/2, bbox2[1] - bbox2[3]/2
    b2_x2, b2_y2 = bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2
    
    # Intersection coordinates
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    # Intersection area
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    union_area = b1_area + b2_area - inter_area
    
    # IoU calculation
    iou = inter_area / union_area if union_area > 0 else 0
    
    return iou

def filter_detections(model_a_file, model_b_file, output_file, iou_threshold=0.2):
    """
    Filter detections from Model A based on Model B's detections
    
    :param model_a_file: Path to Model A's detection results file
    :param model_b_file: Path to Model B's detection results file
    :param output_file: Path to output results file
    :param iou_threshold: Minimum IoU to consider a detection valid
    """
    # Read Model A detections
    with open(model_a_file, 'r') as f:
        model_a_detections = [line.strip().split() for line in f]
    
    # Read Model B detections
    with open(model_b_file, 'r') as f:
        model_b_detections = [line.strip().split() for line in f]
    
    # Group Model B detections by image
    model_b_by_image = {}
    for det in model_b_detections:
        img_name = det[0]
        if img_name not in model_b_by_image:
            model_b_by_image[img_name] = []
        model_b_by_image[img_name].append(det)
    
    # Open output file for writing
    with open(output_file, 'w') as out_f:
        # Process all detections from Model A
        for det in model_a_detections:
            img_name, class_id = det[0], det[1]
            
            # If image is from cam_11 and class_id is 0, validate with IoU
            if 'cam_11' in img_name and class_id == '0':
                # Get corresponding B detections (if any)
                b_dets = model_b_by_image.get(img_name, [])
                
                # Convert detection to bbox format
                a_bbox = [float(det[2]), float(det[3]), float(det[4]), float(det[5])]
                
                # Check if this detection is validated by Model B
                validated = False
                for b_det in b_dets:
                    # Convert B detection to bbox format
                    b_bbox = [float(b_det[2]), float(b_det[3]), float(b_det[4]), float(b_det[5])]
                    
                    # Calculate IoU
                    iou = calculate_iou(a_bbox, b_bbox)
                    
                    # If IoU is above threshold, consider validated
                    if iou >= iou_threshold:
                        validated = True
                        break
                
                # Only write to file if validated
                if validated:
                    out_f.write(' '.join(det) + '\n')
            else:
                # For all other images/classes, write directly
                out_f.write(' '.join(det) + '\n')

# Example usage
filenames = [							
        "fuse_for_cam13_folder/weight_nafgs_withDat_exclude_epoch_3_nafnet.txt",							
        "fuse_for_cam13_folder/weight_nafnet_gsad_epoch_15_nafnet.txt",							
        "fuse_for_cam13_folder/weight_psudo_epoch_4.txt",							
        "fuse_for_cam13_folder/weight_epoch_15.txt",							
        "fuse_for_cam13_folder/weight_nafnet_gsad_8ep_epoch_3_nafnet.txt",							
        "fuse_for_cam13_folder/yolo_nafgs_epoch_60_nafnet.txt",							
        "fuse_for_cam13_folder/pseudo_version5_label_epoch_4.txt",							
        "fuse_for_cam13_folder/yolo_nafgs_pseudo_epoch_55_nafnet.txt",							
        "fuse_for_cam13_folder/yolo_base_exclude_epoch_25.txt",												
    ]

parser = argparse.ArgumentParser(description="Run inference and save predictions.")
parser.add_argument("--path_output", type=str, required=True, help="Path to the input directory containing images.")
args = parser.parse_args()
path_output = f"{args.path_output}/"

for f in filenames:
    filter_detections(f, 
                      'predict_step2/co_dino_5scale_swin_large_16e_o365tococo.txt', 
                      path_output + f.split('/')[-1])
