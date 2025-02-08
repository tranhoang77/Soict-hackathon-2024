import argparse
from ensemble_boxes import weighted_boxes_fusion
import os
from tqdm import tqdm
from pathlib import Path

maxS = 0.35

def read_results_from_txt(filename: Path, thresh=0.075):
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    data = {}
    for line in lines:
        line = line.split(" ")
        img, label, x_center, y_center, w, h, score = line
        x_center, y_center, w, h, score = float(x_center), float(y_center), float(w), float(h), float(score)
        x1, y1, x2, y2 = x_center - w / 2, y_center - h / 2, x_center + w / 2, y_center + h / 2
        if (w * h >= 0.3): continue
        if img not in data:
            data[img] = {}
            data[img]['bboxes'] = []
            data[img]['scores'] = []
            data[img]['labels'] = []
        if (score >= thresh):
            data[img]['bboxes'].append([x1, y1, x2, y2])
            data[img]['scores'].append(score)
            data[img]['labels'].append(label)
        
    return data

def save_wbf_predictions(image_name, boxes, scores, labels, output_file):
    with open(output_file, 'a') as f:
        for box, score, label in zip(boxes, scores, labels):
            # Convert to center, width, height format
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            width = box[2] - box[0]
            height = box[3] - box[1]
            label = int(label)
            
            if (label == 0 and score >= 0.0) or (label != 0 and score >= 0.0):
                line = f"{image_name} {label} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {score:.6f}\n"
                f.write(line)

def parse_arguments():
    """Parse arguments from the command line"""
    parser = argparse.ArgumentParser(description="Fusion predictions with WBF")
    parser.add_argument('--path_private_test', type=str, required=True,
                        help='Path to the public test directory containing images.')
    parser.add_argument('--path_output', type=str, required=True,
                        help='Path to the output dir for fused predictions.')
    parser.add_argument('--path_results', type=str, required=True,
                        help='Base path to the directory containing input result files.')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    public_test = Path(args.path_private_test)
    output_file = f"{args.path_output}/"
    base = f"{args.path_results}/"

    # Các file đầu vào
    filenames = [							
        base + "weight_nafgs_withDat_exclude_epoch_3_nafnet.txt",							
        base + "weight_nafnet_gsad_epoch_15_nafnet.txt",							
        base + "weight_psudo_epoch_4.txt",							
        base + "weight_epoch_15.txt",							
        base + "weight_nafnet_gsad_8ep_epoch_3_nafnet.txt",							
        base + "yolo_nafgs_epoch_60_nafnet.txt",							
        base + "pseudo_version5_label_epoch_4.txt",							
        base + "yolo_nafgs_pseudo_epoch_55_nafnet.txt",							
        base + "yolo_base_exclude_epoch_25.txt",						
    ]
    
    weights = [2,2,2,2,2,1,2,1,1]

    ide = ''.join(str(weights[i]) for i in range(len(weights)))
    output_file += ('ensemble_008private_07_' + ide + "_normal_remove.txt")

    if os.path.exists(output_file):
        os.remove(output_file)
    print(output_file)
    iou_thr = 0.7  # Ngưỡng IoU cho WBF
    skip_box_thr = 0.001  # Confidence threshold để bỏ qua box
    thresh = 0.08

    print("Read data from TXT files")
    # Đọc dữ liệu từ tất cả các file đầu vào
    data_list = [read_results_from_txt(filename, thresh) for filename in filenames]

    print("Processing each image in data")
    for image_path in sorted(public_test.glob("*.jpg")):
        results = [data.get(image_path.name, {'bboxes': [], 'scores': [], 'labels': []}) for data in data_list]

        # Extract boxes, scores, và labels cho WBF
        boxes_list = [result['bboxes'] for result in results]
        scores_list = [result['scores'] for result in results]
        labels_list = [result['labels'] for result in results]

        # Thực hiện WBF nếu có bất kỳ box nào
        if any(boxes_list):
            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                           iou_thr=iou_thr, skip_box_thr=skip_box_thr)

            # Lưu kết quả dự đoán
            save_wbf_predictions(image_path.name, boxes, scores, labels, output_file)

    print("Done")


if __name__ == "__main__":
    main()
