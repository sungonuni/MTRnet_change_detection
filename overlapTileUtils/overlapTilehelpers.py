import logging
import torch

import torch.utils.data
import torch.nn as nn
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)


def decide_crop_number(img_size, crop_size):
    crop_number = 1
    overlap_size = -1

    while(overlap_size < 1):
        crop_number += 1
        overlap_size = (crop_number*crop_size-img_size)/(crop_number-1)
    overlap_size = int(overlap_size)
    
    return crop_number, overlap_size 
        
def split(batch_img, crop_size):
    """
    ------------------
    |      | |       |
    |   0  | |   1   |
    |------|-|-------|
    |------|-|-------|
    |      | |       |
    |   2  | |   3   |
    ------------------
    """
    # input shape : (8,3,512,512)
    #print(batch_img.shape)
    side_start = [] 
    splited_img = torch.empty((0, 3, crop_size, crop_size)) # crop_size = 200
    crop_number, overlap_size = decide_crop_number(batch_img.shape[2], crop_size) # crop_number = 3, overlap_size = 44
    #print(crop_number)
    #print(overlap_size)

    for count in range(crop_number):
        side_start.append(count*(crop_size-overlap_size)) # side_start = [0, 200-44, 400-88]
    #print(side_start)

    for idx in range(batch_img.shape[0]):
        img2split = torch.unsqueeze(batch_img[idx,:,:,:],0) 
        for h in side_start:
            for w in side_start:
                splited_img = torch.cat((splited_img,img2split[:,:,h:h+crop_size,w:w+crop_size]), dim=0)
    
    # print(splited_img.shape)
    return splited_img # splited_img.shape = (72, 3, 200, 200), if batch_size = 8

def split4label(batch_img, crop_size):
    """
    ------------------
    |      | |       |
    |   0  | |   1   |
    |------|-|-------|
    |------|-|-------|
    |      | |       |
    |   2  | |   3   |
    ------------------
    """
    # input shape : (8,3,512,512)
    #print(batch_img.shape)
    side_start = [] 
    splited_img = torch.empty((0, crop_size, crop_size)) # crop_size = 200
    crop_number, overlap_size = decide_crop_number(batch_img.shape[2], crop_size) # crop_number = 3, overlap_size = 44
    #print(crop_number)
    #print(overlap_size)

    for count in range(crop_number):
        side_start.append(count*(crop_size-overlap_size)) # side_start = [0, 200-44, 400-88]
    #print(side_start)

    for idx in range(batch_img.shape[0]):
        img2split = torch.unsqueeze(batch_img[idx,:,:],0) 
        for h in side_start:
            for w in side_start:
                splited_img = torch.cat((splited_img,img2split[:,h:h+crop_size,w:w+crop_size]), dim=0)
    
    #print(splited_img.shape)
    return splited_img # splited_img.shape = (72, 3, 200, 200), if batch_size = 8