import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np

from utils.dataloaders import (full_path_loader, full_test_loader, full_demo_loader, CDDloader)
from adventUtil.discriminator import Discriminator

logging.basicConfig(level=logging.INFO)


def get_source_loaders(opt):

    logging.info('STARTING Dataset Creation')

    train_full_load, _ = full_path_loader(opt.source_dir)


    train_dataset = CDDloader(train_full_load, aug=opt.augmentation)
    # val_dataset = CDDloader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.num_workers)
    
    """
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    """

    return train_loader

def get_target_loaders(opt):

    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.target_dir)


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

def load_discriminator(opt, device):
    device_ids = list(range(opt.num_gpus))

    model = Discriminator(2, 64).to(device)
    model = nn.DataParallel(model, device_ids=device_ids)
    return model

class DLiter():
    def __init__(self, list):
        self.list = list
        self.idx = 0
    def __iter__(self):
        return self
    def rewind(self):
        self.idx = 0
    def __next__(self):
        try:
            return self.list[self.idx]
        except IndexError:
            self.idx = 0
            return self.list[self.idx]
        finally:
            self.idx += 1

def prob_2_entropy(prob):
    b, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob+1e-30)) / np.log2(c)