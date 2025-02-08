from ultralytics import YOLO
import numpy as np
import os
# os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'

model = YOLO("yolo11x.pt")

results = model.train(
    data="training/yolo/naf_gs.yaml", 
    epochs=100, 
    patience=30,
    imgsz=1280, 
    save_period=5,
    batch=2,
    project="naf_gs",
    optimizer='AdamW',
    lr0=0.0001
)


# from pathlib import Path

# # Define the directory containing the .txt files
# f = Path("/mlcv2/WorkingSpace/Personal/quannh/Project/Project/bkai/sample_dataset/dataset_nafnet_split/train/labels")

# # Iterate over sorted .txt files in the directory
# for txt_path in sorted(f.glob("*.txt")):
#     with open(txt_path, 'r') as file:
#         lines = file.readlines()  # Read all lines from the file

#     # Modify each line
#     for i in range(len(lines)):
#         line = lines[i].strip().split(" ")  # Split line into parts and remove whitespace
#         line[0] = str(int(line[0]) % 4)  # Modify the first element
#         lines[i] = ' '.join(line) + '\n'  # Join the modified line back into a string

#     # Print the filename being modified
#     print(txt_path.name)

#     # Write the modified lines back to the file
#     with open(txt_path, 'w') as fi:
#         fi.writelines(lines)  # Write the updated lines back to the file


# Train folder have 12402 images
# Val folder have 3104 images
# CUDA_VISIBLE_DEVICES=2,3 python naf_gs.py