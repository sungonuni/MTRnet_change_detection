import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
from puzzleUtils.puzzleHelpers import (split, merge)

import os
import logging
import json
import sys
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random
import numpy as np


"""
Initialize Parser and define arguments
"""
parser, metadata = get_parser_with_args()
opt = parser.parse_args()

"""
Initialize experiments log
"""
logging.basicConfig(level=logging.INFO)
writer = SummaryWriter(opt.log_dir + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

"""
Set up environment: define paths, download data, and set device
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=777)


train_loader, val_loader = get_loaders(opt)

"""
Load Model or pretrained model
"""
path = opt.pretrain_path

if opt.pretrain_mode:
    try:
        logging.info('Loading pretrained dict model')
        model = load_model(opt, dev)
        model.load_state_dict(torch.load(path), strict=False)
    except:
        logging.info('Changed, Loading dict of full model')
        model = load_model(opt, dev)
        ptmodel = torch.load(path)
        ptmodel_dict = ptmodel.state_dict()
        model.load_state_dict(ptmodel_dict, strict=False)
else:
    logging.info('LOADING vanilla Model')
    model = load_model(opt, dev)

### load full model directly ### 
#device_ids = list(range(opt.num_gpus))
#model = torch.load(path).to(dev)
#model = torch.nn.DataParallel(model, device_ids=device_ids)

"""
Activate loss, optimizer, scheduler function
"""
criterion = get_criterion(opt)
optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

"""
Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    batch_iter = 0
    tbar = tqdm(train_loader)
    for batch_img1, batch_img2, labels in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1

        # Make quarter splited batch img (puzzle module)
        splited_img1 = split(batch_img1)
        splited_img2 = split(batch_img2)

        # Set variables for training
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        labels = labels.long().to(dev)

        splited_img1 = splited_img1.float().to(dev)
        splited_img2 = splited_img2.float().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop (merging module)
        
        entire_preds = model(batch_img1, batch_img2) # entire_preds.shape -> tuple
        splited_preds = model(splited_img1, splited_img2)
        merged_preds = merge(splited_preds) # merged_preds.shape -> torchtensor
        
        # alpha = 0.04 * float(epoch)
        alpha = 1
        
        cd_loss = criterion(entire_preds, merged_preds, labels, alpha) # must set get_criterion to 'puzzle'
        loss = cd_loss
        loss.backward()
        optimizer.step()

        # Calculate and log other batch metrics
        cd_preds = entire_preds[-1]
        _, cd_preds = torch.max(cd_preds, 1)

        cd_corrects = (100 *
                       (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                       (labels.size()[0] * (opt.patch_size**2)))

        cd_train_report = prfs(labels.data.cpu().numpy().flatten(),
                               cd_preds.data.cpu().numpy().flatten(),
                               average='binary',
                               pos_label=1)

        train_metrics = set_metrics(train_metrics,
                                    cd_loss,
                                    cd_corrects,
                                    cd_train_report,
                                    scheduler.get_last_lr())

        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        # clear batch variables from memory
        del batch_img1, batch_img2, splited_img1, splited_img2, labels

    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader:
            # Make quarter splited batch img (puzzle module)
            splited_img1 = split(batch_img1)
            splited_img2 = split(batch_img2)
            
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            splited_img1 = splited_img1.float().to(dev)
            splited_img2 = splited_img2.float().to(dev)

            # Get predictions and calculate loss (merge module)
            entire_preds = model(batch_img1, batch_img2) # entire_preds.shape -> tuple
            splited_preds = model(splited_img1, splited_img2)
            merged_preds = merge(splited_preds) # merged_preds.shape -> torchtensor

            alpha = 0.04 * float(epoch)

            cd_loss = criterion(entire_preds, merged_preds, labels, alpha)

            # Calculate and log other batch metrics
            cd_preds = entire_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            cd_corrects = (100 *
                           (cd_preds.squeeze().byte() == labels.squeeze().byte()).sum() /
                           (labels.size()[0] * (opt.patch_size**2)))

            cd_val_report = prfs(labels.data.cpu().numpy().flatten(),
                                 cd_preds.data.cpu().numpy().flatten(),
                                 average='binary',
                                 pos_label=1)

            val_metrics = set_metrics(val_metrics,
                                      cd_loss,
                                      cd_corrects,
                                      cd_val_report,
                                      scheduler.get_last_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, splited_img1, splited_img2, labels

        logging.info("EPOCH {} VALIDATION METRICS".format(epoch)+str(mean_val_metrics))

        """
        Store the weights of good epochs based on validation results
        """
        if ((mean_val_metrics['cd_precisions'] > best_metrics['cd_precisions'])
                or
                (mean_val_metrics['cd_recalls'] > best_metrics['cd_recalls'])
                or
                (mean_val_metrics['cd_f1scores'] > best_metrics['cd_f1scores'])):

            # Insert training and epoch information to metadata dictionary
            logging.info('updata the model')
            metadata['validation_metrics'] = mean_val_metrics

            # Save model(state_dict) and log
            if not os.path.exists('./tmp'):
                os.mkdir('./tmp')
            with open('./tmp/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model.state_dict(), './tmp/checkpoint_epoch_'+str(epoch)+'.pt')

            # comet.log_asset(upload_metadata_file_path)
            best_metrics = mean_val_metrics


        print('An epoch finished.')
writer.close()  # close tensor board
print('Done!')
