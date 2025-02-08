import argparse
import numpy as np

def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    bbox format: [x_center, y_center, width, height]
    """
    b1_x1, b1_y1 = bbox1[0] - bbox1[2] / 2, bbox1[1] - bbox1[3] / 2
    b1_x2, b1_y2 = bbox1[0] + bbox1[2] / 2, bbox1[1] + bbox1[3] / 2

    b2_x1, b2_y1 = bbox2[0] - bbox2[2] / 2, bbox2[1] - bbox2[3] / 2
    b2_x2, b2_y2 = bbox2[0] + bbox2[2] / 2, bbox2[1] + bbox2[3] / 2

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    union_area = b1_area + b2_area - inter_area

    iou = inter_area / union_area if union_area > 0 else 0

    return iou

def filter_detections(model_a_file, model_b_file, output_file, iou_threshold):
    """
    Filter detections from Model A based on Model B's detections
    """
    with open(model_a_file, 'r') as f:
        model_a_detections = [line.strip().split() for line in f]

    with open(model_b_file, 'r') as f:
        model_b_detections = [line.strip().split() for line in f]

    model_b_by_image = {}
    for det in model_b_detections:
        img_name = det[0]
        if img_name not in model_b_by_image:
            model_b_by_image[img_name] = []
        model_b_by_image[img_name].append(det)

    with open(output_file, 'w') as out_f:
        for det in model_a_detections:
            img_name, class_id = det[0], det[1]
            if 'cam_11' in img_name and class_id == '0':
                b_dets = model_b_by_image.get(img_name, [])
                a_bbox = [float(det[2]), float(det[3]), float(det[4]), float(det[5])]
                validated = False
                for b_det in b_dets:
                    b_bbox = [float(b_det[2]), float(b_det[3]), float(b_det[4]), float(b_det[5])]
                    if calculate_iou(a_bbox, b_bbox) >= iou_threshold:
                        validated = True
                        break
                if validated:
                    out_f.write(' '.join(det) + '\n')
            else:
                out_f.write(' '.join(det) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter detections based on IoU threshold.")
    parser.add_argument("--input_txt", type=str, required=True, help="Path to Model A's detection results file.")
    parser.add_argument("--input_txt_base", type=str, default="predict_step2/co_dino_5scale_swin_large_16e_o365tococo.txt", help="Path to Model B's detection results file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output results file.")
    parser.add_argument("--iou_threshold", type=float, default=0.2, help="Minimum IoU threshold (default: 0.2).")

    args = parser.parse_args()
    filter_detections(args.input_txt, args.input_txt_base, args.output_file, args.iou_threshold)
