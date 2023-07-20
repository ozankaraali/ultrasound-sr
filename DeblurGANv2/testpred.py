import os
from glob import glob
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from fire import Fire
from tqdm import tqdm

from aug import get_normalize
from models.networks import get_generator

import argparse
import os
from torchvision import models, transforms
from torch.autograd import Variable
import shutil

from util.metrics import PSNR
from albumentations import Compose, CenterCrop, PadIfNeeded
from PIL import Image
from ssim.ssimlib import SSIM
from models.networks import get_generator
from glob import glob

class Predictor:
    def __init__(self, config, weights_path):
        # with open('config/config.yaml') as cfg:
        #     config = yaml.load(cfg)
        model = get_generator(config['model'])
        model.load_state_dict(torch.load(weights_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        # GAN inference should be in train mode to use actual stats in norm layers,
        # it's not a bug
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x: np.ndarray, mask: Optional[np.ndarray]):
        x, _ = self.normalize_fn(x, x)
        
        # if mask is None:
        #     mask = np.ones_like(x, dtype=np.float32)
        # else:
        #     mask = np.round(mask.astype('float32') / 255)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {'mode': 'constant',
                      'constant_values': 0,
                      'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
                      }
        x = np.pad(x, **pad_params)
        mask = np.pad(mask, **pad_params)

        return map(self._array_to_batch, (x, mask)), h, w

    @staticmethod
    def _postprocess(x: torch.Tensor) -> np.ndarray:
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img: np.ndarray, mask: Optional[np.ndarray], ignore_mask=True) -> np.ndarray:
        (img, mask), h, w = self._preprocess(img, mask)

        with torch.no_grad():
            inputs = [img.cuda()]
            # if not ignore_mask:
            #     inputs += [mask]
            pred = self.model(*inputs)
            # result_image = pred[0].cpu().float().numpy()
            # result_image = (np.transpose(result_image, (1, 2, 0)) + 1) / 2.0 * 255.0
            # result_image = result_image.astype('uint8')
            
            # gt_image = mask.cpu().float().numpy()
            # gt_image = np.squeeze(gt_image)
            # gt_image = (np.transpose(gt_image, (1, 2, 0)) + 1) / 2.0 * 255.0
            # gt_image = gt_image.astype('uint8')

            # psnr = PSNR(result_image, gt_image)
            # pilFake = Image.fromarray(result_image)
            # pilReal = Image.fromarray(gt_image)
            # ssim = SSIM(pilFake).cw_ssim_value(pilReal)
            # return psnr, ssim
        return self._postprocess(pred)[:h, :w, :]#, psnr, ssim

def main(config_path='config/config.yaml'):
    with open(config_path, 'r',encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    model_name = 'best_{}.h5'.format(config['experiment_desc'])
    predictor = Predictor(config=config, weights_path=model_name)

    # create output dir which is like 'results/{experiment_desc}'

    # create if results dir does not exist
    if not os.path.exists('results'):
        os.mkdir('results')

    # create experiment_desc dir inside of results dir
    experiment_dir = os.path.join('results', config['experiment_desc'])
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)

    # datasets:
    #     test_1:  # the 1st test dataset
    #         name: CCA-test
    #         type: PairedImageDataset
    #         dataroot_gt: /home/karaaliozan/dataset/CCA/test/HR/x4
    #         dataroot_lq: /home/karaaliozan/dataset/CCA/test/Bic/x4
    #         io_backend:
    #         type: disk

    tests = [
        ('CCA-test', '/home/karaaliozan/dataset/CCA/test/HR/x4', '/home/karaaliozan/dataset/CCA/test/Bic/x4'),
        ('CCABlur-test', '/home/karaaliozan/dataset/CCA/test/HRBlur/x4', '/home/karaaliozan/dataset/CCA/test/LRBlurBic/x4'),
        ('BUSI-test', '/home/karaaliozan/dataset/BUSI/test/HR/x4', '/home/karaaliozan/dataset/BUSI/test/Bic/x4'),
        ('BUSIBlur-test', '/home/karaaliozan/dataset/BUSI/test/HRBlur/x4', '/home/karaaliozan/dataset/BUSI/test/LRBlurBic/x4'),
        ('CCANoise-test', '/home/karaaliozan/dataset/CCA/test/HRBlurNoise/x4', '/home/karaaliozan/dataset/CCA/test/LRBlurNoiseBic/x4'),
        ('BUSINoise-test', '/home/karaaliozan/dataset/BUSI/test/HRBlurNoise/x4', '/home/karaaliozan/dataset/BUSI/test/LRBlurNoiseBic/x4'),
    ]

    for i in range(len(tests)):
        test1_out_dir = os.path.join(experiment_dir, tests[i][0])
        if not os.path.exists(test1_out_dir):
            os.mkdir(test1_out_dir)

        # get all images in the test dataset lq
        test1_lq_dir = tests[i][2]
        test1_lq_imgs = glob(os.path.join(test1_lq_dir, '*'))

        # get all images in the test dataset gt
        test1_gt_dir = tests[i][1]
        test1_gt_imgs = glob(os.path.join(test1_gt_dir, '*'))

        # sort the images by name
        test1_lq_imgs.sort()
        test1_gt_imgs.sort()

        # iterate over the images
        avg_psnr = 0
        avg_ssim = 0
        for j, (f_img, f_gt) in tqdm(enumerate(zip(test1_lq_imgs, test1_gt_imgs))):
            name = os.path.basename(f_img)
            img, gt = map(cv2.imread, (f_img, f_gt))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)

            pred = predictor(img, gt)
            pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)

            psnr = PSNR(pred.astype('uint8'), gt.astype('uint8'))
            ssim = SSIM(Image.fromarray(pred.astype('uint8'))).cw_ssim_value(Image.fromarray(gt.astype('uint8')))
            avg_psnr += psnr
            avg_ssim += ssim
            cv2.imwrite(os.path.join(test1_out_dir, name), pred)
        
        avg_psnr = avg_psnr / len(test1_lq_imgs)
        avg_ssim = avg_ssim / len(test1_lq_imgs)
        
        # report as text file results.txt:
        with open(os.path.join(test1_out_dir, 'results.txt'), 'w') as f:
            f.write('PSNR: {:.4f} SSIM: {:.4f}'.format(avg_psnr, avg_ssim))

        # also print experiment results in a format: experiment, current test, psnr, ssim
        print(config['experiment_desc'])
        print(tests[i][0])
        print('{:.4f}, {:.4f}'.format(avg_psnr, avg_ssim))


if __name__ == '__main__':
    Fire(main)