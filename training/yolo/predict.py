import argparse
from pathlib import Path
from ultralytics import YOLO

LABELS = {
    0: 'bike',
    1: 'car',
    2: 'coach',
    4: 'container'
}

def extract_data_batch(batch_path: Path, model_weight: Path, output_file: Path, confidence_threshold: float = 0.1):  
    # Load model
    model = YOLO(model_weight)
    
    # Create destination directory
    dest_dir = output_file.parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Delete output file if it already exists
    if output_file.exists():
        output_file.unlink()
        print(f"Deleted existing file: {output_file}")
    
    # Process images in batch
    for image_path in sorted(batch_path.glob("*.jpg")):
        predictions = model(image_path, save_txt=None)
        
        with open(output_file, 'a') as file:    
            for idx, prediction in enumerate(predictions[0].boxes.xywhn):
                cls = int(predictions[0].boxes.cls[idx].item())
                score = float(predictions[0].boxes.conf[idx].item()) 
                
                if score >= confidence_threshold:
                    file.write(f"{image_path.name} {cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()} {score}\n")

def main():
    # Parse arguments from terminal
    parser = argparse.ArgumentParser(description="Batch processing for YOLO predictions.")
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the directory containing image batch.")
    parser.add_argument('--checkpoint_file', type=str, required=True, help="Path to the YOLO model weight file.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to save the output predictions.")
    parser.add_argument('--confidence_threshold', type=float, default=0.1, help="Confidence threshold for predictions (default: 0.1).")

    args = parser.parse_args()

    # Call the function with parsed arguments
    extract_data_batch(
        batch_path=Path(args.input_dir),
        model_weight=Path(args.checkpoint_file),
        output_file=Path(args.output_file),
        confidence_threshold=args.confidence_threshold
    )

if __name__ == "__main__":
    main()
