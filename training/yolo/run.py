import argparse
from ultralytics import YOLO

def main(args):
    model = YOLO('yolo11x.pt')

    results = model.train(
        data=args.path_file_yaml,
        epochs=args.epochs,
        patience=args.patience,
        project =args.project,
        imgsz=args.imgsz,
        device=args.device,
        batch=args.batch,
        save_period=args.save_period,
    )
    print("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO model with custom parameters")
    parser.add_argument("--path_file_yaml", type=str, required=True, help="Path to data.yaml file")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Patience for early stopping")
    parser.add_argument("--project", type=str, default="weight", help="store checkpoint")
    parser.add_argument("--imgsz", type=int, default=1280, help="Image size")
    parser.add_argument("--device", type=str, default="0,1", help="Device to use for training (e.g., '0', '0,1')")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--save_period", type=int, default=5, help="Save model every N epochs")
    
    args = parser.parse_args()
    main(args)
