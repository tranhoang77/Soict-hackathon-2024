import os
from pathlib import Path
from collections import defaultdict
import numpy as np

# Hàm tính toán IoU
def calculate_iou(bbox1, bbox2):
    ix_min = max(bbox1[0], bbox2[0])
    iy_min = max(bbox1[1], bbox2[1])
    ix_max = min(bbox1[2], bbox2[2])
    iy_max = min(bbox1[3], bbox2[3])

    if ix_min >= ix_max or iy_min >= iy_max:
        return 0

    intersection_area = (ix_max - ix_min) * (iy_max - iy_min)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area

# Đọc kết quả từ file văn bản
def read_results_from_txt(filename: Path, thresh=0.075):
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} không tồn tại.")
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {}
    for line in lines:
        line = line.split(" ")
        img, label, x_center, y_center, w, h, score = line
        x_center, y_center, w, h, score = float(x_center), float(y_center), float(w), float(h), float(score)
        x1, y1, x2, y2 = x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2
        if (w * h >= 0.3): continue  # Bỏ qua các bounding box nhỏ
        if img not in data:
            data[img] = {'bboxes': [], 'scores': [], 'labels': []}
        if (score >= thresh):
            data[img]['bboxes'].append([x1, y1, x2, y2])
            data[img]['scores'].append(score)
            data[img]['labels'].append(label)
        
    return data

# Theo dõi và thêm đối tượng vào file trung gian
def track_and_add_objects(input_file, target_image, threshold_percentage=0.25, iou_threshold=0.75, intermediate_file="intermediate.txt"):
    data = read_results_from_txt(input_file)
    target_bboxes = data.get(target_image, {}).get('bboxes', [])
    target_labels = data.get(target_image, {}).get('labels', [])
    
    if not target_bboxes:
        print(f"No bounding boxes found for {target_image}")
        with open(intermediate_file, 'w') as f:
            pass  # Tạo file rỗng nếu không có bounding box
        return
    
    all_cam_11_images = [img for img in data.keys() if 'cam_11' in img]
    total_images = len(all_cam_11_images)
    object_tracking = defaultdict(int)
    
    for image_name in all_cam_11_images:
        if image_name == target_image:
            continue
        bboxes = data.get(image_name, {}).get('bboxes', [])
        labels = data.get(image_name, {}).get('labels', [])
        for i, (target_bbox, target_label) in enumerate(zip(target_bboxes, target_labels)):
            for bbox, label in zip(bboxes, labels):
                if target_label == label:
                    iou = calculate_iou(target_bbox, bbox)
                    if iou >= iou_threshold:
                        object_tracking[(target_label, tuple(target_bbox))] += 1

    with open(intermediate_file, 'w') as f:
        with open(input_file, 'r') as in_file:
            f.writelines(in_file.readlines())
        
        for (label, bbox), count in object_tracking.items():
            if count / total_images >= threshold_percentage:
                for image_name in all_cam_11_images:
                    existing = False
                    bboxes = data.get(image_name, {}).get('bboxes', [])
                    for existing_bbox in bboxes:
                        if calculate_iou(bbox, existing_bbox) >= iou_threshold + 0.1:
                            existing = True
                            break
                    if not existing:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                        x_center, y_center = (x1 + x2) / 2, (y1 + y2) / 2
                        f.write(f"{image_name} {label} {x_center} {y_center} {w} {h} 0.3\n")
    print(f"Intermediate file saved to {intermediate_file}")

# Non-max suppression
def non_max_suppression(detections, iou_threshold=0.6):
    detections = sorted(detections, key=lambda x: x[-1], reverse=True)
    selected_detections = []

    while detections:
        current = detections.pop(0)
        selected_detections.append(current)
        detections = [
            box for box in detections
            if calculate_iou(current[2:6], box[2:6]) <= iou_threshold
        ]
    return selected_detections

# Xử lý file để loại bỏ bounding box chồng lấn
def process_file(input_file, output_file, iou_threshold=0.6):
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} không tồn tại.")
    
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    detections = []
    for line in lines:
        parts = line.strip().split()
        image_name, cls, x, y, w, h, confidence = parts[0], int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
        detections.append((image_name, cls, x, y, w, h, confidence))

    grouped_detections = defaultdict(list)
    for det in detections:
        key = (det[0], det[1])
        grouped_detections[key].append(det)

    final_detections = []
    for key, dets in grouped_detections.items():
        final_detections.extend(non_max_suppression(dets, iou_threshold))

    with open(output_file, "w") as f:
        for det in final_detections:
            f.write(f"{det[0]} {det[1]} {det[2]} {det[3]} {det[4]} {det[5]} {det[6]:.6f}\n")
    print(f"Final output saved to {output_file}")

# Main function
def main(input_file, final_output_file, target_image, threshold_percentage=0.25, iou_threshold_tracking=0.75, iou_threshold_nms=0.6):
    intermediate_file = "intermediate.txt"
    
    print("Step 1: Tracking and adding objects...")
    track_and_add_objects(
        input_file=input_file, 
        target_image=target_image,
        threshold_percentage=threshold_percentage, 
        iou_threshold=iou_threshold_tracking, 
        intermediate_file=intermediate_file
    )
    
    print("Step 2: Removing overlapping bounding boxes...")
    process_file(
        input_file=intermediate_file, 
        output_file=final_output_file, 
        iou_threshold=iou_threshold_nms
    )
    
    print(f"Processing completed. Results saved to {final_output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Track and deoverlap bounding boxes from an input file.")
    
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input file containing bounding box data.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to save the final output file.")
    parser.add_argument("-img", "--target_image", type=str, default='cam_11_00385.jpg', help="Path to the input file containing bounding box data.")
    parser.add_argument("-tp", "--threshold_percentage", type=float, default=0.25, help="Threshold percentage for object tracking (default: 0.25).")
    parser.add_argument("-tiou", "--tracking_iou", type=float, default=0.75, help="IoU threshold for tracking objects (default: 0.75).")
    parser.add_argument("-diou", "--deoverlap_iou", type=float, default=0.6, help="IoU threshold for non-max suppression (default: 0.6).")
    
    args = parser.parse_args()
    
    main(
        input_file=args.input,
        final_output_file=args.output,
        target_image=args.target_image,
        threshold_percentage=args.threshold_percentage,
        iou_threshold_tracking=args.tracking_iou,
        iou_threshold_nms=args.deoverlap_iou
    )
