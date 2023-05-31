import torch
import torchvision.models as models
import torch.nn.functional as F
from torch import nn
import math
import os
import numpy as np
import yaml
import argparse
import datetime

from utilities.model_helper import build_model
from utilities.dataloader_helper import build_dataloader
from utilities.tester_helper import Tester
from utilities.utils_helper import create_logger

def main():

    cfg = {
    'random_seed': 444,
    'dataset': {
        'type': 'KITTI',
        'batch_size': 4,
        'use_3d_center': True,
        'class_merging': False,
        'use_dontcare': False,
        'bbox2d_type': 'anno',
        'meanshape': False,
        'writelist': ['Car'],
        'random_flip': 0.5,
        'random_crop': 0.5,
        'scale': 0.4,
        'shift': 0.1
    },
    'model': {
        'type': 'centernet3d',
        'backbone': 'dla34',
        'neck': 'DLAUp',
        'num_class': 3
    },
    'optimizer': {
        'type': 'adam',
        'lr': 0.00125,
        'weight_decay': 1e-05
    },
    'lr_scheduler': {
        'warmup': True,
        'decay_rate': 0.1,
        'decay_list': [90, 120]
    },
    'trainer': {
        'max_epoch': 140,
        'gpu_ids': '0',
        'save_frequency': 10,
        'pretrained' :False
    },
    'tester': {
        'type': 'KITTI',
        'mode': 'single',
        'checkpoint': '/Users/strom/Desktop/test_giou_double_pretrained/checkpoint_epoch_140.pth', #use this line to load the desired pretrained model
        'checkpoints_dir': 'checkpoints',
        'threshold': 0.2
    },
    'evaluate': True,
}

    log_file = 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = create_logger(log_file)

    # build dataloader
    _ , test_loader  = build_dataloader(cfg['dataset'])

    # build model
    model = build_model(cfg['model'])


    if cfg['evaluate']:
            tester = Tester(cfg=cfg['tester'],
                            model=model,
                            dataloader=test_loader,
                            logger=logger)
            tester.test()


if __name__ == '__main__':
    main()