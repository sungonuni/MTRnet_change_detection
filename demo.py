'''
This file is used to save the demo image
'''
import logging
import sys
import torch.utils.data
from utils.parser import get_parser_with_args
from utils.helpers import get_demo_loaders, initialize_metrics, load_model
import os
from tqdm import tqdm
import cv2

if not os.path.exists('./output_img'):
    os.mkdir('./output_img')

parser, metadata = get_parser_with_args()
opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_loader = get_demo_loaders(opt, batch_size=1)


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

model.eval()
index_img = 0
test_metrics = initialize_metrics()
with torch.no_grad():
    tbar = tqdm(test_loader)
    for batch_img1, batch_img2, labels in tbar:

        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        cd_preds = model(batch_img1, batch_img2)
        cd_preds = cd_preds[-1] # model returns only tuple, so take [-1] to get tensor 
        #print("new")
        #print(cd_preds)
        _, cd_preds = torch.max(cd_preds, 1)
        #print(cd_preds) 
        cd_preds = cd_preds.data.cpu().numpy()
        cd_preds = cd_preds.squeeze() * 255

        file_path = './output_img/' + str(index_img).zfill(5)
        cv2.imwrite(file_path + '.png', cd_preds)

        index_img += 1
