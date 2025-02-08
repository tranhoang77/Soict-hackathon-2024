from pathlib import Path
from ultralytics import YOLO
import argparse

LABELS = {
    0: 'bike',
    1: 'car',
    2: 'coach',
    4: 'container'
}

def extract_data_batch(model_name: str, batch_path: Path, predict_file_name: str, output_folder: str):
    model = YOLO(model_name)
    dest_dir = Path(output_folder)
    predict_file_path = dest_dir / predict_file_name
    if predict_file_path.exists():
        predict_file_path.unlink()  # Deletes the file
        print(f"Deleted existing file: {predict_file_path}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for image_path in sorted(batch_path.glob("*.jpg")):
        predictions = model(image_path, save_txt=None)
        
        with open(predict_file_path, 'a') as file:
            for idx, prediction in enumerate(predictions[0].boxes.xywhn):
                cls = int(predictions[0].boxes.cls[idx].item())
                score = float(predictions[0].boxes.conf[idx].item()) 
                
                if score >= 0.1:
                    file.write(f"{image_path.name} {cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()} {score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data batch using YOLO model")
    parser.add_argument("--path_private_test", type=str, required=True, help="Path to the directory containing image batch")
    parser.add_argument("--path_private_test_nafgs", type=str, required=True, help="Path to the directory containing image batch with nafnet")
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    extract_data_batch(
        model_name = "weights/epoch55.pt",
        batch_path = Path(args.path_private_test_nafgs),
        predict_file_name = "yolo_nafgs_pseudo_epoch_55_nafnet.txt",
        output_folder = args.output_folder
    )
    extract_data_batch(
        model_name = "weights/epoch60.pt",
        batch_path = Path(args.path_private_test_nafgs),
        predict_file_name = "yolo_nafgs_epoch_60_nafnet.txt",
        output_folder = args.output_folder
    )
    extract_data_batch(
        model_name = "weights/epoch25.pt",
        batch_path = Path(args.path_private_test),
        predict_file_name = "yolo_base_exclude_epoch_25.txt",
        output_folder = args.output_folder
    )
