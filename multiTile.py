import datetime
import torch
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)
from multiTileUtils.multiTileHelpers import (get_main_loaders, get_aux1_loaders, get_aux2_loaders)
import os
import logging
import json
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
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
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


train_loader_512, val_loader_512 = get_main_loaders(opt)
train_loader_256 = get_aux1_loaders(opt)
train_loader_128 = get_aux2_loaders(opt)
lambda_512 = opt.lambda_main
lambda_256 = opt.lambda_aux1
lambda_128 = opt.lambda_aux2

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
    tbar = tqdm(train_loader_512)
    train_loader_256_iter = iter(train_loader_256)
    train_loader_128_iter = iter(train_loader_128)

    for batch512_img1, batch512_img2, labels_512 in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(batch_iter) + " - " + str(batch_iter+opt.batch_size))
        batch_iter = batch_iter+opt.batch_size
        total_step += 1

        # Set 512 tiles for training
        batch512_img1 = batch512_img1.float().to(dev)
        batch512_img2 = batch512_img2.float().to(dev)
        labels_512 = labels_512.long().to(dev)

        # Set 256 tiles for training
        batch256 = train_loader_256_iter.__next__()
        batch256_img1, batch256_img2, labels_256 = batch256
        batch256_img1 = batch256_img1.float().to(dev)
        batch256_img2 = batch256_img2.float().to(dev)
        labels_256 = labels_256.long().to(dev)

        # Set 128 tiles for training
        batch128 = train_loader_128_iter.__next__()
        batch128_img1, batch128_img2, labels_128 = batch128
        batch128_img1 = batch128_img1.float().to(dev)
        batch128_img2 = batch128_img2.float().to(dev)
        labels_128 = labels_128.long().to(dev)

        # Zero the gradient
        optimizer.zero_grad()

        # Get model predictions, calculate loss, backprop

        cd_preds_512 = model(batch512_img1, batch512_img2)
        cd_loss_512 = criterion(cd_preds_512, labels_512)

        cd_preds_256 = model(batch256_img1, batch256_img2)
        cd_loss_256 = criterion(cd_preds_256, labels_256)

        cd_preds_128 = model(batch128_img1, batch128_img2)
        cd_loss_128 = criterion(cd_preds_128, labels_128)

        loss = lambda_512*cd_loss_512 + lambda_256*cd_loss_256 + lambda_128*cd_loss_128
        loss.backward()
        optimizer.step()

        cd_preds = cd_preds_512[-1]
        labels = labels_512
        cd_loss = loss
        _, cd_preds = torch.max(cd_preds, 1)

        # Calculate and log other batch metrics
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
        del (
            batch512_img1, batch512_img2, labels_512,
            batch256_img1, batch256_img2, labels_256,
            batch128_img1, batch128_img2, labels_128
                )

    scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    """
    Begin Validation
    """
    model.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels in val_loader_512:
            # Set variables for training
            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss
            cd_preds = model(batch_img1, batch_img2)

            cd_loss = criterion(cd_preds, labels)

            cd_preds = cd_preds[-1]
            _, cd_preds = torch.max(cd_preds, 1)

            # Calculate and log other batch metrics
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

            for k, v in mean_val_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del batch_img1, batch_img2, labels

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
