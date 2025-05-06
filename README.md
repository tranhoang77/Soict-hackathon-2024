# Object Detection Project Guide

## 1. Environment Setup

README: Workspace Setup Guide
This guide explains how to set up the workspace by unzipping and organizing files into the appropriate directories.

1. Steps to Set Up the Workspace
   Unzip the main file:

Extract the main zip file to access its contents, which include multiple zip files.
Extract `data.zip`folders to the data directory:
Unzip all folders found in the main file into the data directory.
Extract files from folders:

Unzip the files inside the extracted folders from the previous step into the root directory.
Extract training-specific files:

Navigate to the `training` folder.
Unzip the `yolo.zip`, `Co-DETR_base.zip` and `Co-DETR.zip` files found in this folder.

Delete all zip files after extraction is complete.

2. Final Workspace Structure
   The workspace should follow this structure after all steps are completed:

```css
workspace/
├── data/
│   ├── pseudo_2e_label_on_train
│   ├── public_test
│   ├── ...
│
├── training/
│   ├── YOLO/
│   │   ├── [YOLO Files]
│   ├── Co-DETR/
│   │   ├── [Co-DETR Files]
│   ├── Co-DETR_base/
│   │   ├── [Co-DETR_base Files]

│
├── inference
|    ├── [Files]
├── preprocessing
|    ├── [Files]

└── README.md
```

### Docker Setup

1. Pull Docker Image

```bash
docker pull haeako/rookie_uit_bkai:v2
```

2. Run Docker Container (example)

```bash
docker run -itd --ipc=host --gpus=all haeako/rookie_uit_bkai:v2 /bin/bash
```

### Build-in Conda Environments

| Environment      | Purpose        | Activation Command      |
| ---------------- | -------------- | ----------------------- |
| `YOLO`           | YOLO Models    | `conda activate YOLO`   |
| `nafnet`         | NAF-Net Models | `conda activate nafnet` |
| `gsad`           | GSAD Models    | `conda activate gsad`   |
| `base` (Default) | Co-DETR Models | `conda activate base`   |

## Overview

We use 10 checkpoint files with a unique approach of fusing results from different model configurations.
This project involves object detection using multiple models and techniques, including:

- Co-DETR
- YOLO
- NAFNet
- GSAD

### Data Processing Strategy

#### Image Types

- Original images
- Processed images (using NAFNET and GSAD)

#### Label Types

- Original labels
- Pseudo-generated labels on the training set

## 2. Data Pre-processing

### 2.1 Pseudo Label Generation

```bash
cd Co-DETR
python3 pseudo_add_label.py --image_dir /path/to/train_image_dir --output_folder_label /path/to/folder_save_label
```

### 2.2 Create COCO Format for Co-DETR

- The data folder to train will appear in the Co-DETR folder.
- delete_cam05 is optional

```bash
cd preprocessing
python3 codetr_format_train.py --train_dir_image /path/to/data_train_dir_image --train_dir_label /path/to/data_train_dir_label --output_dir /path/to/training/Co-DETR
```

### 2.3 Create COCO Format for YOLO

- The data folder to train will appear in the yolo folder.
- delete_cam05 is optional

```bash
cd preprocessing
python3 yolo_format_train.py --train_dir_image /path/to/data_train_dir_image --train_dir_label /path/to/data_train_dir_label --output_dir /path/to/training/yolo
```

### 2.4 NAF-Net Image Processing Script

This script supports image processing using the NAFNet model for two main functions:

- Deblurring
- Denoising

### Usage

```bash
conda acitvate nafnet
cd preprocessing/NAFNet
```

```bash
python inference.py \
    --mode [deblur/denoise] \
    --train_input /path/to/train_data \
    --public_test_input /path/to/public_data
```

### Output Results

The script automatically generates new output folders with a `_nafnet` suffix in a same base folder ( e.x `data/train` will generate `data/train_nafnet`), which contains processed images and corresponding grounth truth files.

#### Example Output

```
data/train_nafnet/
├── image1.jpg
├── image1.txt
├── image2.jpg
└── image2.txt
```

### 2.5 GSAD Image Processing Script:

#### Usage

```bash
conda activate gsad
cd preprocessing/GSAD/
```

```bash
python inference.py \
    --input /path/to/input/images \
    --output /path/to/output/images \
    -gpu [GPU_id]
```

Key Arguments
`--input:` Directory containing input images
`--output:` Directory to save processed images
`-gpu:` GPU device ID to use (default: 0)

Output Characteristics
Processed image named with `_gsad` suffix when image is not in public_test
Preserves original image dimensions
Copies corresponding .txt files if present

Example
Process training images;

```bash
python inference.py \
    --input ../../data/train_nafnet \
    --output ../../data/train_nafnet \
    -gpu 0
```

```bash
python inference.py \
    --input ../../data/public_test_nafnet \
    --output ../../data/public_test_nafnet_gsad \
    -gpu 0
```

## 3. Training

- For model training, each checkpoint file will be trained from different data (specific information about data requirements of each checkpoint file can be found in "Training Configurations for Each Checkpoint"). To create the data format, please run "Create COCO Format for Co-DETR/YOLO" with appropriate parameters for each checkpoint file requirement.
- Note: you need to change the config file to accommodate the new number of classes instead of the 80 classes in the coco dataset

### Co-DETR Training (2 GPUs)

```bash
conda activate base
cd Co-DETR
sh tools/dist_train.sh
```

### YOLO Training (2 GPUs)

```bash
conda activate YOLO
cd yolo
python3 run.py --path_file_yaml /path/to/data/data.yaml
```

## 4. Inference

If the model checkpoint contains the keyword `nafgs`, the script will use test data from an alternate directory:

Default test data path:
`data/public_test/`

Special case path: `data/public_test_nafnet_gsad/`

### 4.1 Co-DETR Prediction

- If you want to run predict corresponding to each checkpoint file.
- Note: if checkpoint file use NAFNET, you will use public_test_nafnet_gsad.

```bash
cd Co-DETR
python3 predict.py --checkpoint_file /path/to/checkpoint_file --input_dir /path/to/public_test_folder --output_file /path/to/inference/results/file.txt
```

#### Run with shell file for multiple checkpoint files:

- Overview:
  You can replace checkpoint file to predict in .sh file.
  This script manages parallel prediction tasks across multiple GPUs. It processes multiple checkpoints and generates corresponding output files.

- for models use NAF-net prepocessed data

```bash
bash predict_nafgs.sh [GPU_list]
```

- for moldes use normal data:

```bash
bash predict.sh [GPU_list]
```

- Arguments

GPU_list: One or more GPU IDs (required)

Example: `bash predict.sh 0 1 2` to use GPUs 0, 1, and 2

- Input Configuration

The script uses two predefined arrays:

`CHECKPOINTS`: List of model checkpoint paths

`OUTPUTS`: List of output file paths
Prediction Results

Saves prediction results in specified output files trong thư mục `inference/test_result`

### 4.2 YOLO Prediction

- Run with python file:
- If you want to run predict corresponding to each checkpoint file.
- Note: if checkpoint file use NAFNET, you will use public_test_nafnet_gsad.

```bash
cd training/yolo
python3 predict.py --checkpoint_file /path/to/checkpoint_file --input_dir /path/to/public_test_folder --output_file /path/to/inference/results/file.txt
```

#### Run with shell file for multiple checkpoint files:

(Check 4.1 for more detail)

- for models use NAF-net prepocessed data

```bash
bash predict_nafgs.sh [GPU_list]
```

example: `bash predict_nafgs.sh 0 1 2 3`

- for moldes use normal data:

```bash
bash predict.sh [GPU_list]
```

example: `bash predict.sh 0 1 2 3`

### 4.3 Model Fusion

- Note: path_result is the path to the folder containing .txt result files. Run code in (base) env
- file submit.txt will be created in folder_to_save

```bash
cd inference
python3 fuse_withtxt.py --path_public_test /path/to/public_test --output_file /path/to/folder_to_save --path_results /path/to/inference/results
```

## Training Configurations for Each Checkpoint

### Co-DETR (8 Configurations)

1. Epoch 4: Original images + Processing labels
2. Epoch 15: Original images + Original labels
3. Epoch 15: Processing images (cam 05 deleted) + Original labels
4. Epoch 3: Processing images + Original labels
5. Epoch 2: Processing images (cam 05 deleted) + Processing labels
6. Epoch 15: Original images + Processing labels
7. Epoch 15: Processing images + Original labels
8. Epoch 10: Processing images (cam 05 deleted) + Original labels

### YOLO (3 Configurations)

1. Epoch 25: Original images (cam 05 deleted) + Original labels
2. Epoch 55: Processing images + Processing labels
3. Epoch 60: Processing images + Original labels

# 5. Predict for private test

- For each camera we will use a separate detection method.

## 5.1 For cam_11

- We inference the prediction results through 5 steps

### 5.1.1.

- We will use two checkpoint files: epoch 10 when training with Processing images (cam 05 deleted) + Original labels; epoch 15 when training normally with the organizer's label set. The two checkpoint files we have trained are saved in the "Submit_BKAI/training/Co-DETR/weight/weight_private_test" folder in Co-DETR.
- (Check 4.1 for more detail to predict)
- The result after running predict as above will be 2 .txt files and they can be saved as in the folder "Submit_BKAI/inference/predict_step1"

### 5.1.2.

- I use pre-trained model "co_dino_5scale_swin_large_16e_o365tococo.pth" a model in https://github.com/Sense-X/Co-DETR to detect motorcycle

```bash
cd training/Co-DETR_base
CUDA_VISIBLE_DEVICES=[GPU_ID] python3 predict.py --checkpoint_file models/co_dino_5scale_swin_large_16e_o365tococo.pth --input_dir /path/to/private_test --output_file /path/to/inference/predict_step2/co_dino_5scale_swin_large_16e_o365tococo.txt
```

### 5.1.3.

- Use inference/fuse_to_remove.py to remove walking person who was misidentified as a motorcycle.
- The result after running predict as above will be 2 .txt files and they can be saved as in the folder "Submit_BKAI/inference/predict_step3_fuse_base_to_remove_person"

```bash
cd inference
python3 remove_person.py --input_txt /path/to/predict_step1/file_txt_to_remove --input_txt_base /path/to/inference/predict_step2/co_dino_5scale_swin_large_16e_o365tococo.txt --output_file /path/to/inference/predict_step3_fuse_base_to_remove_person/file_name.txt
```

- we recommend "file_name.txt" should be "nafgs_10e_01_remove_person_015.txt" and "pseudo_15e_01_remove_person_015.txt"

### 5.1.4 Fusing model

**Weighted Boxes Fusion (WBF) Pipeline**
This script processes bounding box predictions by applying Weighted Boxes Fusion (WBF) to combine predictions from multiple input sources. The fused output file contains refined bounding boxes for each image, ensuring higher accuracy and consistency.

- Features

1. **Fusion of Predictions**: Combines bounding boxes from multiple prediction sources using a weighted approach.
2. **Confidence Thresholding**: Filters out low-confidence boxes to improve the quality of the output.
3. **Bounding Box Adjustment**: Converts bounding box coordinates to the required center, width, and height format.

- Usage
  Running the Script
  Run the script using the following command:
- Run code in (base) env

```bash
cd inference/
python fuse_2txt.py --path_data_test /path/to/private_test --output_folder /path/to/predict_step4_folder
```

Required Arguments:
`--path_data_test`: Path to the directory containing the public test images in .jpg format.
`--output_folder`: Path to the directory where the output file will be saved.
we recommend using `predict_step4` as the output folder

- Result:
  The fused output file will be saved in the specified output directory, with a name similar to:

```bash
predict_step4/nafgs_10e_withDat_01_remove_person_015-pseudo_15e_01_remove_015_1-1_iou-0.55_skip_box-0.2_1-16-2-2_5-1-4-25.txt
```

- Notes:
  **Weights Configuration**: Adjust the weights variable in the script to specify the importance of each input file.

**Thresholds\***: You can modify iou_thr (IoU threshold) and skip_box_thr (confidence threshold) in the script to control the behavior of the fusion process.

**Intermediate Processing\***: The script dynamically processes all .jpg images in the provided public test directory.
**Ensure that file `inference/predict_step3_fuse_base_to_remove_person/nafgs_10e_01_remove_person_015.txt` and
`inference/predict_step3_fuse_base_to_remove_person/pseudo_15e_01_remove_person_015.txt` existed , if not please check previous step**

### 5.1.5 Tracking

**Bounding Box Tracking and Deoverlap Pipeline**

This script processes bounding box data by performing **tracking** and **non-max suppression (deoverlap)** to produce a cleaned output file. This code is only applied for camera ID 'cam_11'

### Features

1. **Tracking**: Identifies bounding boxes that consistently appear across multiple images based on Intersection over Union (IoU).
2. **Deoverlap**: Removes overlapping bounding boxes using non-max suppression.

### Usage

#### Running the Script

To use the script, run it via the command line with the following arguments:

```bash
cd inference
python tracking.py -i /path/to/predict_step4/input_file.txt -o /path/to/predict_step5/output_file.txt
```

- Optional Arguments
  `-tp`, --threshold_percentage: Threshold percentage for tracking objects (default: 0.25).
  `-tiou`, --tracking_iou: IoU threshold for tracking objects (default: 0.75).
  `-diou`, --deoverlap_iou: IoU threshold for non-max suppression (default: 0.6).

- Example:

```bash
python tracking.py \
-i input.txt \
-o output.txt \
-tp 0.3 \
-tiou 0.8 \
-diou 0.5
```

- Output File Format:
  The output file will be in the same format as the input file but with processed bounding boxes.
- Intermediate File:
  An intermediate file (intermediate.txt) will be created during processing. It can be ignored or removed after the script finishes.

## 5.2 Model fusion to detect on camera 13

### Inference and Post-Processing Steps

### 5.2.1. Inference Co-DETRs

```bash
cd training/Co-DETR
conda activate base
CUDA_VISIBLE_DEVICES=0 python predict_totxt.py --path_private_test /path/to/private_test --path_private_test_nafgs /path/to/private_test_nafnet --output_dir /path/to/inference/fuse_for_cam13_folder
```

### 5.2.2. Inference YOLO

```bash
cd training/yolo
conda activate YOLO
CUDA_VISIBLE_DEVICES=1 python inference.py --path_private_test /path/to/private_test --path_private_test_nafgs /path/to/private_test_nafnet --output_folder /path/to/inference/fuse_for_cam13_folder
```

### 5.2.3. Fuse to Remove Person

```bash
cd inference
conda activate base
python fuse_remove.py --path_output /path/to/fuse_remove
```

### 5.2.4. Fuse All Results

```bash
cd inference
python fuse_withtxt.py --path_private_test /path/to/private_test --path_results /path/to/fuse_remove --path_output /path/to/fuse_all
```

## 5.3

- When you have the .txt file results from camera 11 and camera 13, run the code setscore.py to get the final submit file.

```bash
cd inference
python3 setscore.py --input_file_cam11 /path/to/predict_step5/tracking.txt --input_file_cam13 /path/to/fuse_all/ensemble_008private_07_222221211_normal_remove.txt --output_file /path/to/submit_file_txt
```
