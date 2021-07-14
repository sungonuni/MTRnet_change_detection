import sys
import datetime
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as prfs
from utils.parser import get_parser_with_args
from utils.helpers import (get_loaders, get_criterion,
                           load_model, initialize_metrics, get_mean_metrics,
                           set_metrics)

from adventUtil.adventHelpers import (get_source_loaders, get_target_loaders, 
                                    load_discriminator, prob_2_entropy, DLiter)
from adventUtil.adventLoss import bce_loss

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
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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

# Should set opt.source_dir, opt.target_dir
s_train_loader = get_source_loaders(opt)
t_train_loader, t_val_loader = get_target_loaders(opt)

#train_loader, val_loader = get_loaders(opt)

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

logging.info('LOADING vanilla discriminator')
discriminator = load_discriminator(opt, dev)

"""
Activate loss, optimizer, scheduler function
"""
criterion = get_criterion(opt)
d_loss_fn = bce_loss

m_optimizer = torch.optim.AdamW(model.parameters(), lr=opt.learning_rate) # Be careful when you adjust learning rate, you can refer to the linear scaling rule
m_scheduler = torch.optim.lr_scheduler.StepLR(m_optimizer, step_size=8, gamma=0.5)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(0.9, 0.99))
d_scheduler = torch.optim.lr_scheduler.StepLR(d_optimizer, step_size=8, gamma=0.5)

"""
Set starting values
"""
best_metrics = {'cd_f1scores': -1, 'cd_recalls': -1, 'cd_precisions': -1}
logging.info('STARTING training')
total_step = -1

source_label = 0
target_label = 1

# set s_train to DLiter
t_train_loader_iter = enumerate(t_train_loader)

for epoch in range(opt.epochs):
    train_metrics = initialize_metrics()
    val_metrics = initialize_metrics()

    """
    Begin Training
    """
    model.train()
    logging.info('SET model mode to train!')
    discriminator.train()
    logging.info('SET discriminator mode to train!')

    source_iter = 0    
    tbar = tqdm(s_train_loader)


    for source_img1, source_img2, s_labels in tbar:
        tbar.set_description("epoch {} info ".format(epoch) + str(source_iter) + " - " + str(source_iter+opt.batch_size))
        source_iter = source_iter+opt.batch_size
        total_step += 1
        m_loss = 0
        d_loss = 0

        source_img1 = source_img1.float().to(dev)
        source_img2 = source_img2.float().to(dev)
        s_labels = s_labels.long().to(dev)
        
        # Test required
        try:
            _, t_batch = t_train_loader_iter.__next__()
        except:
            t_train_loader_iter = enumerate(t_train_loader)
            _, t_batch = t_train_loader_iter.__next__()

        target_img1, target_img2, t_labels = t_batch
        target_img1 = target_img1.float().to(dev)
        target_img2 = target_img2.float().to(dev)
        t_labels = t_labels.long().to(dev)

        m_optimizer.zero_grad()
        d_optimizer.zero_grad()
        
        # Turn off discriminator grad backprop
        for param in discriminator.parameters():
            param.requires_grad = False
        for param in model.parameters(): # added to experiment
            param.requires_grad = True

        # Train CD model (source -> source)
        s_predict = model(source_img1, source_img2)
        m_cd_loss = criterion(s_predict, s_labels)
        m_loss = m_cd_loss

        m_loss.backward()


        # Adversarial training (target -> source)
        t_predict = model(target_img1, target_img2)
        t_predict = t_predict[-1]
        t_d_out = discriminator(prob_2_entropy(F.softmax(t_predict, dim=1)))
        print(t_predict.shape)
        print(F.softmax(t_predict, dim=1).shape)
        print(prob_2_entropy(F.softmax(t_predict, dim=1)).shape)
        print(t_d_out.shape)
        m_adv_loss = d_loss_fn(t_d_out, source_label)
        m_loss = 0.001 * m_adv_loss

        m_loss.backward()

        # Turn on discriminator grad backprop
        for param in discriminator.parameters():
            param.requires_grad = True
        for param in model.parameters(): # added to experiment
            param.requires_grad = False

        # Train discriminator (source -> source)
        s_predict = s_predict[-1]
        s_predict = s_predict.detach()
        s_d_out = discriminator(prob_2_entropy(F.softmax(s_predict, dim=1)))
        d_s_loss = d_loss_fn(s_d_out, source_label)
        d_loss = 0.5 * d_s_loss

        d_loss.backward()

        # Train discriminator (target -> target)
        t_predict = t_predict.detach()
        t_d_out = discriminator(prob_2_entropy(F.softmax(t_predict, dim=1)))
        d_t_loss = d_loss_fn(t_d_out, target_label)
        d_loss = 0.5 * d_t_loss

        d_loss.backward()

        for param in discriminator.parameters():
            param.requires_grad = True
        for param in model.parameters(): # added to experiment
            param.requires_grad = True

        m_optimizer.step()
        d_optimizer.step()

        _, cd_preds = torch.max(t_predict, 1)
        labels = t_labels
        cd_loss = m_loss

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
                                    m_scheduler.get_last_lr())
        
        # log the batch mean metrics
        mean_train_metrics = get_mean_metrics(train_metrics)

        for k, v in mean_train_metrics.items():
            writer.add_scalars(str(k), {'train': v}, total_step)

        del source_img1, source_img2, s_labels, target_img1, target_img2, t_labels
    
    m_scheduler.step()
    d_scheduler.step()
    logging.info("EPOCH {} TRAIN METRICS".format(epoch) + str(mean_train_metrics))

    """
    Begin Validation
    """

    model.eval()
    with torch.no_grad():
        for target_img1, target_img2, labels in t_val_loader:
            # Set variables for training
            target_img1 = target_img1.float().to(dev)
            target_img2 = target_img2.float().to(dev)
            labels = labels.long().to(dev)

            # Get predictions and calculate loss
            cd_preds = model(target_img1, target_img2)

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
                                      m_scheduler.get_last_lr())

            # log the batch mean metrics
            mean_val_metrics = get_mean_metrics(val_metrics)

            for k, v in mean_train_metrics.items():
                writer.add_scalars(str(k), {'val': v}, total_step)

            # clear batch variables from memory
            del target_img1, target_img2, labels

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
            if not os.path.exists('./tmp4advent'):
                os.mkdir('./tmp4advent')
            with open('./tmp4advent/metadata_epoch_' + str(epoch) + '.json', 'w') as fout:
                json.dump(metadata, fout)

            torch.save(model.state_dict(), './tmp4advent/checkpoint_epoch_'+str(epoch)+'.pt')

            # comet.log_asset(upload_metadata_file_path)
            best_metrics = mean_val_metrics


        print('An epoch finished.')
writer.close()  # close tensor board
print('Done!')
