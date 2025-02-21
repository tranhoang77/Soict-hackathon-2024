import argparse
import torch
import shutil
from basicsr.models import create_model
from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
from basicsr.utils.options import parse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img

def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

def single_image_inference(model, img, save_path):
      model.feed_data(data={'lq': img.unsqueeze(dim=0)})

      if model.opt['val'].get('grids', False):
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      sr_img = tensor2img([visuals['result']])
      cv2.imwrite(save_path, sr_img)

def main():
    parser = argparse.ArgumentParser(description='NAFNet Image Inference')
    parser.add_argument('--mode', type=str, choices=['deblur', 'denoise'], 
                        default='deblur', 
                        help='Choose between deblur and denoise modes')
    parser.add_argument('--train_input', type=str, 
                        default="../data/train",
                        help='Input directory for training data')
    parser.add_argument('--public_test_input', type=str, 
                        default='../data/public_test', 
                        help='Input directory for public test data')
    
    args = parser.parse_args()

    # Chọn đường dẫn option dựa trên mode
    if args.mode == 'deblur':
        opt_path = 'options/test/REDS/NAFNet-width64.yml'
    else:  # denoise
        opt_path = 'options/test/SIDD/NAFNet-width64.yml'

    opt = parse(opt_path, is_train=False)
    opt['dist'] = False
    NAFNet = create_model(opt)

    # Xử lý dữ liệu train
    INPUT = Path(args.train_input)
    OUTPUT = Path(str(INPUT) + "_nafnet")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(INPUT.glob("*.jpg")):
        print(img_path.name)
        img_input = imread(str(img_path))
        inp = img2tensor(img_input)
        single_image_inference(NAFNet, inp, str(OUTPUT / img_path.name))
        txt_file = INPUT / (img_path.stem + ".txt")
        if txt_file.exists():
            shutil.copy(txt_file, OUTPUT)
    
    # Xử lý public test
    INPUT = Path(args.public_test_input)
    OUTPUT = Path(str(INPUT) + "_nafnet")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    for img_path in sorted(INPUT.glob("*.jpg")):
        print(img_path.name)
        img_input = imread(str(img_path))
        inp = img2tensor(img_input)
        single_image_inference(NAFNet, inp, str(OUTPUT / img_path.name))

if __name__ == "__main__":
    main()