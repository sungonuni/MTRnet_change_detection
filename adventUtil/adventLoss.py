import torch
import torch.nn as nn

def bce_loss(y_pred, y_label):
    y_label_tensor = torch.FloatTensor(y_pred.size())
    y_label_tensor.fill_(y_label)
    y_label_tensor = y_label_tensor.to(y_pred.get_device())
    return nn.BCEWithLogitsLoss()(y_pred, y_label_tensor)