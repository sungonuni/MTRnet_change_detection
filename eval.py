import torch.utils.data
import os
import sys
import logging
from utils.parser import get_parser_with_args
from utils.helpers import get_test_loaders, load_model
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# The Evaluation Methods in our paper are slightly different from this file.
# In our paper, we use the evaluation methods in train.py. specifically, batch size is considered.
# And the evaluation methods in this file usually produce higher numerical indicators.

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_test_loaders(opt)

path = opt.pretrain_path

if opt.pretrain_mode:
    try:
        logging.info('Loading dict model')
        model = load_model(opt, dev)
        model.load_state_dict(torch.load(path), strict=False)
    except:
        logging.info('Changed, Loading dict of full model')
        model = load_model(opt, dev)
        ptmodel = torch.load(path)
        ptmodel_dict = ptmodel.state_dict()
        model.load_state_dict(ptmodel_dict, strict=False)
else:
    sys.exit('Please set pretrain_mode to True!')
    
"""
path = './results/SNUnet-32(SCD256)/SNUnet-32(SCD256).pt'   # the path of the model
# model = torch.load(path)                # CDData: torch.load / SCData: load_state_dict 
print(path)
model = load_model(opt, dev)
model.load_state_dict(torch.load(path))
"""

c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
model.eval()

with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        tn, fp, fn, tp = confusion_matrix(labels.data.cpu().numpy().flatten(),
                        cd_preds.data.cpu().numpy().flatten()).ravel()

        c_matrix['tn'] += tn
        c_matrix['fp'] += fp
        c_matrix['fn'] += fn
        c_matrix['tp'] += tp

tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
P = tp / (tp + fp)
R = tp / (tp + fn)
F1 = 2 * P * R / (R + P)

print('Precision: {}\nRecall: {}\nF1-Score: {}'.format(P, R, F1))
