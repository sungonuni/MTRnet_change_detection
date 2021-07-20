import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

from utils.dataloaders import (full_path_loader, full_test_loader, full_demo_loader, CDDloader)
from adventUtil.discriminator import Discriminator

logging.basicConfig(level=logging.INFO)

def get_main_loaders(opt):

    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dir_main)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader

def get_aux1_loaders(opt):

    logging.info('STARTING Dataset Creation')

    train_full_load, _ = full_path_loader(opt.dir_aux1)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=int(opt.batch_size)*4,
                                               shuffle=True,
                                               num_workers=opt.num_workers)

    return train_loader

def get_aux2_loaders(opt):

    logging.info('STARTING Dataset Creation')

    train_full_load, _ = full_path_loader(opt.dir_aux2)

    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=int(opt.batch_size)*16,
                                               shuffle=True,
                                               num_workers=opt.num_workers)

    return train_loader



