import os
import logging
import pickle
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
#import torchvision

from IPython.core.debugger import set_trace
#import matplotlib.pyplot as plt

from model import AtlasResNet34, AtlasResNet18
from model2 import AtlasInceptionV3
from model3 import AtlasBNInception

from utils import save_checkpoint, load_checkpoint, set_logger
from dataset import prepare_trainset
from gpu_utils import *

######### Define the training process #########
def run_check_net(train_dl, val_dl, multi_gpu=[0, 1]):
    set_logger(LOG_PATH)
    logging.info('\n\n')
    #---
    if MODEL == 'RESNET34':
        net = AtlasResNet34(debug=False).cuda(device=device)
    elif MODEL == 'RESNET18':
        net = AtlasResNet18(debug=False).cuda(device=device)
    elif MODEL == 'INCEPTION_V3':
        net = AtlasInceptionV3(debug=False, num_classes=28, aux_logits=AUX_LOGITS, transform_input=False).cuda(device=device)
    elif MODEL == 'BNINCEPTION':
        net = AtlasBNInception(debug=False, num_classes=28).cuda(device=device)

#     for param in net.named_parameters():
#         if param[0][:8] in ['decoder5']:#'decoder5', 'decoder4', 'decoder3', 'decoder2'
#             param[1].requires_grad = False

    # dummy sgd to see if it can converge ...
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=LearningRate, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.045)#LearningRate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=4,#4 resnet34 
                                                           verbose=False, threshold=0.0001, 
                                                           threshold_mode='rel', cooldown=0, 
                                                           min_lr=0, eps=1e-08)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.9, last_epoch=-1)
    
    if warm_start:
        logging.info('warm_start: '+last_checkpoint_path)
        net, _ = load_checkpoint(last_checkpoint_path, net)
    
    # using multi GPU
    if multi_gpu is not None:
        net = nn.DataParallel(net, device_ids=multi_gpu)

    diff = 0
    best_val_metric = 0.0
    optimizer.zero_grad()
    
    #seed = get_seed()
    #seed = SEED
    #logging.info('aug seed: '+str(seed))
    #ia.imgaug.seed(seed)
    #np.random.seed(seed)
    
    for i in range(NUM_EPOCHS):
        t0 = time.time()
        # iterate through trainset
        if multi_gpu is not None:
            net.module.set_mode('train')
        else:
            net.set_mode('train')
        train_loss_list, train_metric_list = [], []
        #for seed in [1]:#[1, SEED]:#augment raw data with a duplicate one (augmented)
        #seed = get_seed()
        #np.random.seed(seed)
        #ia.imgaug.seed(i//10)
        for input_data, truth in train_dl:
            #set_trace()
            input_data, truth = input_data.to(device=device, dtype=torch.float), \
            truth.to(device=device, dtype=torch.float)
            logit = net(input_data)#[:, :3, :, :]
            
            if multi_gpu is not None:
                _train_loss  = net.module.criterion(logit, truth)
                _train_metric  = net.module.metric(logit, truth)
            else:
                _train_loss  = net.criterion(logit, truth)
                _train_metric  = net.metric(logit, truth)
            train_loss_list.append(_train_loss.detach())
            train_metric_list.append(_train_metric)#.detach()

            _train_loss.backward()#_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss = np.mean(train_loss_list)
        train_metric = np.mean(train_metric_list)

        # compute valid loss & iou (for memory efficiency, use batch)
        net.module.set_mode('valid')
        with torch.no_grad():
            val_loss_list, val_metric_list = [], []
            #input_data_valid, truth_valid = None, None
            for input_data, truth in val_dl:
                input_data, truth = input_data.to(device=device, dtype=torch.float), \
                truth.to(device=device, dtype=torch.float)#device=device,device='cpu'
                logit = net(input_data)
                
                if multi_gpu is not None:
                    _val_loss = net.module.criterion(logit, truth)
                    _val_metric = net.module.metric(logit, truth)
                else:
                    _val_loss = net.criterion(logit, truth)
                    _val_metric = net.metric(logit, truth)
                val_loss_list.append(_val_loss)
                val_metric_list.append(_val_metric)
            val_loss = np.mean(val_loss_list)
            val_metric = np.mean(val_metric_list)
            #if multi_gpu is not None:
            #    val_loss = net.module.criterion(logit, truth_valid)
            #    val_metric = net.module.metric(logit, truth_valid)
            #else:
            #    val_loss = net.criterion(logit, truth_valid)
            #    val_metric = net.metric(logit, truth_valid)

        # Adjust learning_rate
        scheduler.step(val_metric)
        #
        if val_metric > best_val_metric:
            best_val_metric = val_metric
            is_best = True
            diff = 0
        else:
            is_best = False
            diff += 1
            if diff > early_stopping_round:
                logging.info('Early Stopping: val_iou does not increase %d rounds'%early_stopping_round)
                #print('Early Stopping: val_iou does not increase %d rounds'%early_stopping_round)
                break
        
        #save checkpoint
        checkpoint_dict = \
        {
            'epoch': i,
            'state_dict': net.module.state_dict() if multi_gpu is not None else net.state_dict(),
            'optim_dict' : optimizer.state_dict(),
            'metrics': {'train_loss': train_loss, 'val_loss': val_loss, 'train_iou': train_metric, 'val_iou': val_metric}
#             'metrics': {'train_loss1': train_loss1, 
#                         'val_loss1': val_loss1, 
#                         'train_iou1': train_iou1, 
#                         'val_iou1': val_iou1}
        }
        save_checkpoint(checkpoint_dict, is_best=is_best, checkpoint=checkpoint_path)

        #if i%20==0:
        if i>-1:
            #logging.info('[EPOCH %05d][mask coverage zero] train_loss, train_iou: %0.5f, %0.5f; val_loss, val_iou: %0.5f, %0.5f'%(i, train_loss0.item(), train_iou0.item(), val_loss0.item(), val_iou0.item()))
            logging.info('[EPOCH %05d][all classes] train_loss, train_metric: %0.5f, %0.5f; val_loss, val_metric: %0.5f, %0.5f; time elapsed: %0.1f min'%(i, train_loss.item(), train_metric.item(), val_loss.item(), val_metric.item(), (time.time()-t0)/60))
            #logging.info('[EPOCH %05d] train_loss, train_iou: %0.5f,%0.5f; val_loss, val_iou: %0.5f,%0.5f'%(i, train_loss.item(), train_iou.item(), val_loss.item(), val_iou.item()))
        i = i+1


######### Config the training process #########
#device = set_n_get_device("0, 1, 2, 3", data_device_id="cuda:0")#0, 1, 2, 3, IMPORTANT: data_device_id is set to free gpu for storing the model, e.g."cuda:1"
MODEL = 'RESNET34'#'RESNET34', 'RESNET18', 'INCEPTION_V3', 'BNINCEPTION'
#AUX_LOGITS = True#False, only for 'INCEPTION_V3'
print('====MODEL ACHITECTURE: %s===='%MODEL)

device = set_n_get_device("0, 1, 2, 3", data_device_id="cuda:0")#0, 1, 2, 3, IMPORTANT: data_device_id is set to free gpu for storing the model, e.g."cuda:1"
multi_gpu = [0, 1, 2]#use 2 gpus

SEED = 1234#5678#4567#3456#2345#1234
debug = True# if True, load 100 samples
BATCH_SIZE = 32#64 for 256x256, 32 for 512x512
NUM_WORKERS = 20
warm_start, last_checkpoint_path = False, 'checkpoint/%s_v1_seed%s/best.pth.tar'%(MODEL, SEED)
checkpoint_path = 'checkpoint/%s_256_v1_seed%s'%(MODEL, SEED)#resnet34: 1222_v2_seed%s, inception_v3: 1227_v1_seed%s, 
LOG_PATH = 'logging/%s_256_v1_seed%s.log'%(MODEL, SEED)#
torch.cuda.manual_seed_all(SEED)

NUM_EPOCHS = 100
early_stopping_round = 10#500#50
LearningRate = 0.02#phase1: 0.02, phase2: 0.005

######### Load Atlas Protein data #########
# for v1
train_dl, val_dl = prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, debug, 
                                    sample_mode='raw', use_sampler=True)#raw,balance, weak_balance

# for v2
#train_dl, val_dl = prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, debug, 
#                                    sample_mode='balance', use_sampler=False)#raw,balance, weak_balance

######### Run the training process #########
run_check_net(train_dl, val_dl, multi_gpu=multi_gpu)