from utils.parser import get_parser_with_args
from utils.metrics import FocalLoss, dice_loss, EP_loss

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

def hybrid_loss(predictions, target):
    """Calculating the loss"""
    loss = 0

    # gamma=0, alpha=None --> CE
    focal = FocalLoss(gamma=0, alpha=None)

    for prediction in predictions:

        bce = focal(prediction, target)
        dice = dice_loss(prediction, target)
        loss += bce + dice

    return loss

"""
for EPL experiment
"""
def hybrid_epl_loss(predictions, target):
    """Calculating the loss"""
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

