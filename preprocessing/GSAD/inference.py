import os
from os.path import basename
import math
import argparse
import random
import logging
import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image
import options.options as option
from utils import util
import torchvision.transforms as T
import model as Model
import core.logger as Logger
import core.metrics as Metrics
import natsort
from torchvision import transforms
from pathlib import Path
import shutil

transform = transforms.Lambda(lambda t: (t * 2) - 1)

GPU = "5"

def main():
    #### options
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to option YMAL file.',
                            default='./config/dataset.yml') 
    parser.add_argument('--input', type=str, help='testing the unpaired image')
    parser.add_argument('--output', type=str, help='testing the unpaired image', default="")
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--tfboard', action='store_true')
    parser.add_argument('-c', '--config', type=str, default='config/test_unpaired.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                        help='Run either train(training) or val(generation)', default='train')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=GPU)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_wandb_ckpt', action='store_true')
    parser.add_argument('-log_eval', action='store_true')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    opt['phase'] = 'test'

    #### distributed training settings
    opt['dist'] = False
    rank = -1
    print('Disabled distributed training.')

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        # config loggers. Before it, the log will not work
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))


    util.setup_logger('base', opt['path']['log'], 'train', level=logging.INFO, screen=True)
    logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    #### random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')
    
    InputPath = Path(args.input)
    if args.output: OutputPath = Path(args.output)
    else: OutputPath = Path(opt['path']['results'])
    OutputPath.mkdir(parents=True, exist_ok=True)
    isTest = False
    if 'test' in OutputPath.name: isTest = True

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')

    for img_path in sorted(InputPath.glob("*.jpg")):
        if len(img_path.stem) < 20:
            txt_file = InputPath / (img_path.stem + ".txt")
            if txt_file.exists():
                shutil.copy(txt_file, OutputPath)
            shutil.copy(img_path, OutputPath)
            continue
        
        if not isTest:
            txt_file = InputPath / (img_path.stem + ".txt")
            if txt_file.exists():
                shutil.copy(txt_file, OutputPath)
            shutil.copy(img_path, OutputPath)

        raw_img = Image.open(img_path).convert('RGB')
        img_w = raw_img.size[0]
        img_h = raw_img.size[1]
        raw_img = transforms.Resize((img_h // 16 * 16, img_w // 16 * 16))(raw_img)

        raw_img = transform(TF.to_tensor(raw_img)).unsqueeze(0).cuda()

        val_data = {}
        val_data['LQ'] = raw_img
        val_data['GT'] = raw_img
        diffusion.feed_data(val_data)
        diffusion.test(continous=False)

        visuals = diffusion.get_current_visuals()
        
        normal_img = Metrics.tensor2img(visuals['HQ']) 
        normal_img = cv2.resize(normal_img, (img_w, img_h))

        llie_img_mode = 'single'
        if llie_img_mode == 'single':
            if isTest:
                util.save_img(
                    normal_img, str(OutputPath / (img_path.stem + ".jpg")))
            else:
                util.save_img(
                    normal_img, str(OutputPath / (img_path.stem + "_gsad.jpg")))
                txt_file = InputPath / (img_path.stem + ".txt")
                if txt_file.exists():
                    shutil.copy(txt_file, OutputPath / (img_path.stem + "_gsad.txt"))


if __name__ == '__main__':
    main()
