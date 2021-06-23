import logging
import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
import sys

logging.basicConfig(level=logging.INFO)

def split(batch_img):
    """
    -----------------
    |       |       |
    |   0   |   2   |
    |       |       |
    -----------------
    |       |       |
    |   1   |   3   |
    |       |       |
    -----------------
    """
    # input shape : (16,3,512,512)
    for idx in range(batch_img.shape[0]):
        if idx == 0:
            img2split = batch_img[idx,:,:,:]
            img2split = torch.unsqueeze(img2split, 0) # shape : (1,3,512,512)
            vertical_split0, vertical_split1 = torch.tensor_split(img2split, 2, dim=3)

            split_chunk0, split_chunk1 = torch.tensor_split(vertical_split0, 2, dim=2)
            split_chunk2, split_chunk3 = torch.tensor_split(vertical_split1, 2, dim=2)

            splited_img = torch.cat((split_chunk0, split_chunk1, split_chunk2, split_chunk3), 0) # shape : (4,3,256,256)
        else:
            img2split = batch_img[idx,:,:,:]
            img2split = torch.unsqueeze(img2split, 0)
            vertical_split0, vertical_split1 = torch.tensor_split(img2split, 2, dim=3)

            split_chunk0, split_chunk1 = torch.tensor_split(vertical_split0, 2, dim=2)
            split_chunk2, split_chunk3 = torch.tensor_split(vertical_split1, 2, dim=2)

            splited_img = torch.cat((splited_img, split_chunk0, split_chunk1, split_chunk2, split_chunk3), 0)

    # output shape : (64,3,256,256)
    return splited_img

def merge(splited_preds):
    splited_preds = splited_preds[-1] # input shape : (64,2,256,256)
    for idx in range(splited_preds.shape[0])[::4]:
        if idx == 0:
            img2concat = splited_preds[idx:idx+4,:,:,:] # shape : (4,2,256,256)
            vertical_concat0 = torch.cat((img2concat[0,:,:,:], img2concat[1,:,:,:]), 1)
            vertical_concat1 = torch.cat((img2concat[2,:,:,:], img2concat[3,:,:,:]), 1)
            merged_pred = torch.cat((vertical_concat0, vertical_concat1), 2)
            merged_pred = torch.unsqueeze(merged_pred, 0) # shape : (1,2,512,512)
        else:
            img2concat = splited_preds[idx:idx+4,:,:,:]
            vertical_concat0 = torch.cat((img2concat[0,:,:,:], img2concat[1,:,:,:]), 1)
            vertical_concat1 = torch.cat((img2concat[2,:,:,:], img2concat[3,:,:,:]), 1)
            merged_1layer = torch.cat((vertical_concat0, vertical_concat1), 2)
            merged_1layer = torch.unsqueeze(merged_1layer, 0)
            merged_pred = torch.cat((merged_pred, merged_1layer), 0)

    # output shape : (16,2,512,512)
    return merged_pred
