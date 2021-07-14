import torch.nn as nn
from utils.parser import get_parser_with_args
from utils.losses import hybrid_loss

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def puzzle_loss(e_preds, m_preds, target, alpha): # e_preds : tuple, m_preds : tensor
    
    e_loss = hybrid_loss(e_preds, target)
    # print("e_loss: "+str(e_loss))

    m_preds = (m_preds, ) # change m_preds to tuple
    m_loss = hybrid_loss(m_preds, target)
    # print("m_loss: "+str(m_loss))

    e_preds = e_preds[-1] # change e_preds to tensor
    m_preds = m_preds[0]  # change m_preds to tensor
    recons_loss = nn.L1Loss()(e_preds, m_preds)
    # print("recons_loss: "+str(recons_loss))
    
    total_loss = e_loss + m_loss + (alpha * recons_loss)
    # print("total_loss: "+str(total_loss))

    return total_loss  

"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target):
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

def hybrid_epl_loss(predictions, target):
    loss = 0
    lmbd = 0.5

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        epl = EP_loss(prediction, target)

        # print("bce"+ str(bce))
        # print("dice"+ str(dice))
        # print("epl"+ str(epl))
        loss += bce + dice + (lmbd * epl)

    return loss
"""