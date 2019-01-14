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

from model_with_mask import AtlasResNet34

from utils import save_checkpoint, load_checkpoint, set_logger
from dataset import prepare_trainset_withMask
from gpu_utils import *

######### Define the training process #########
def run_check_net(train_dl, val_dl, multi_gpu=[0, 1]):
    set_logger(LOG_PATH)
    logging.info('\n\n')
    #---
    net = AtlasResNet34(debug=False).cuda(device=device)

    for param in net.named_parameters():#do not backward ['decoder', 'center'] layers params
        if 'decoder' in param[0]:
            param[1].requires_grad = False
            #print('skipping layer: ', param[0])
        elif 'center' in param[0]:
            param[1].requires_grad = False
            #print('skipping layer: ', param[0])
        elif 'resnet' in param[0] and param[0]:#only backward fc layers, not in ['resnet.fc.weight', 'resnet.fc.bias']
            param[1].requires_grad = False
            #print('skipping layer: ', param[0])
        elif 'encoder' in param[0]:
            param[1].requires_grad = False
            #print('skipping layer: ', param[0])
    print('='*20)
    print('Number of layers that requires_grad=True: ', len(list(filter(lambda p: p.requires_grad, net.parameters()))))

    # dummy sgd to see if it can converge ...
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                      lr=LearningRate, momentum=0.9, weight_decay=0.0001)
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.045)#LearningRate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                           factor=0.5, patience=2,#4 resnet34 
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
        #train_loss_list, train_metric_list = [], []
        train_loss_clf_list, train_loss_mask_list = [], []
        train_metric_list = []
        for input_data, truth in train_dl:
            #set_trace()
            input_data, truth = input_data.to(device=device, dtype=torch.float), truth.to(device=device, dtype=torch.float)
            image, mask = input_data[:, :3, :, :], input_data[:, 3, :, :].view(-1, 1, 512, 512)#[:, 4, :, :]
            logit_clf, logit_mask = net(image)
            
            if multi_gpu is not None:
                _train_loss_clf, _train_loss_mask = net.module.criterion(logit_clf, truth, logit_mask, mask)#_train_loss
                _train_metric  = net.module.metric(logit_clf, truth, device='gpu')
            else:
                _train_loss_clf, _train_loss_mask  = net.criterion(logit_clf, truth, logit_mask, mask)#_train_loss
                _train_metric  = net.metric(logit_clf, truth, device='gpu')
            #train_loss_list.append(_train_loss.detach())
            train_loss_clf_list.append(_train_loss_clf.detach())
            train_loss_mask_list.append(_train_loss_mask.detach())
            train_metric_list.append(_train_metric)#.detach()
            
            _train_loss = 0.9 * _train_loss_clf + 0.1 * _train_loss_mask###borrow from model_with_mask.py
            _train_loss.backward()#_train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        #train_loss = np.mean(train_loss_list)
        train_loss_clf, train_loss_mask = np.mean(train_loss_clf_list), np.mean(train_loss_mask_list)
        train_metric = np.mean(train_metric_list)

        # compute valid loss & metrics (concatenate valid set in cpu, then compute loss, metrics on full valid set)
        net.module.set_mode('valid')
        with torch.no_grad():
            logit_valid, truth_valid = None, None
            for input_data, truth in val_dl:
                input_data = input_data.to(device=device, dtype=torch.float)
                image = input_data[:, :3, :, :]#[:, :4, :, :]                
                logit = net(image)
                logit = logit.cpu().float()#.numpy()#.detach()
                truth = truth.cpu().float()
                if logit_valid is None:
                    logit_valid = logit
                    truth_valid = truth
                else:
                    logit_valid = torch.cat((logit_valid, logit), dim=0)
                    truth_valid = torch.cat((truth_valid, truth), dim=0)
            if multi_gpu is not None:
                val_loss = net.module.criterion(logit_valid, truth_valid)
                val_metric = net.module.metric(logit_valid, truth_valid, device='cpu')
            else:
                val_loss = net.criterion(logit_valid, truth_valid)
                val_metric = net.metric(logit_valid, truth_valid, device='cpu')

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
            'metrics': {'train_loss_clf': train_loss_clf, 
                        'train_loss_mask': train_loss_mask,
                        'val_loss': val_loss, 
                        'train_iou': train_metric, 'val_iou': val_metric}
#             'metrics': {'train_loss1': train_loss1, 
#                         'val_loss1': val_loss1, 
#                         'train_iou1': train_iou1, 
#                         'val_iou1': val_iou1}
        }
        save_checkpoint(checkpoint_dict, is_best=is_best, checkpoint=checkpoint_path)

        #if i%20==0:
        if i>-1:
            #logging.info('[EPOCH %05d][mask coverage zero] train_loss, train_iou: %0.5f, %0.5f; val_loss, val_iou: %0.5f, %0.5f'%(i, train_loss0.item(), train_iou0.item(), val_loss0.item(), val_iou0.item()))
            logging.info('[EPOCH %05d][all classes] (train_loss_clf, train_loss_mask, train_loss_total), train_metric: (%0.5f, %0.5f, %0.5f), %0.5f; val_loss_clf, val_metric: %0.5f, %0.5f; time elapsed: %0.1f min'%(i, train_loss_clf.item(), train_loss_mask.item(), (0.9*train_loss_clf.item()+0.1*train_loss_mask.item()), train_metric.item(), val_loss.item(), val_metric.item(), (time.time()-t0)/60))
            #logging.info('[EPOCH %05d] train_loss, train_iou: %0.5f,%0.5f; val_loss, val_iou: %0.5f,%0.5f'%(i, train_loss.item(), train_iou.item(), val_loss.item(), val_iou.item()))
        i = i+1

######### Config the training process #########
#device = set_n_get_device("0, 1, 2, 3", data_device_id="cuda:0")#0, 1, 2, 3, IMPORTANT: data_device_id is set to free gpu for storing the model, e.g."cuda:1"
MODEL = 'RESNET34_MASK'
#AUX_LOGITS = True#False, only for 'INCEPTION_V3'
print('====MODEL ACHITECTURE: %s===='%MODEL)

device = set_n_get_device("0, 1, 2, 3", data_device_id="cuda:3")#0, 1, 2, 3, IMPORTANT: data_device_id is set to free gpu for storing the model, e.g."cuda:1"
multi_gpu = [3, 1, 2]#use 2 gpus

SEED = 1234#5678#4567#3456#2345#1234
debug = False# if True, load 100 samples
IMG_SIZE = 512#256
BATCH_SIZE = 32#64 for 256x256, 32 for 512x512
NUM_WORKERS = 20
warm_start, last_checkpoint_path = True, 'checkpoint/RESNET34_MASK_512_v1_seed1234/best.pth.tar'
checkpoint_path = 'checkpoint/%s_%s_v2_seed%s'%(MODEL, IMG_SIZE, SEED)
LOG_PATH = 'logging/%s_%s_v2_seed%s.log'%(MODEL, IMG_SIZE, SEED)#
torch.cuda.manual_seed_all(SEED)

NUM_EPOCHS = 100
early_stopping_round = 10#500#50
LearningRate = 0.002#phase1: 0.02, phase2: 0.002

######### Load Atlas Protein data #########
# for v1
#train_dl, val_dl = prepare_trainset_withMask(BATCH_SIZE, NUM_WORKERS, SEED, debug, 
#                                    sample_mode='raw', use_sampler=True, IMG_SIZE=IMG_SIZE)#raw,balance, weak_balance

# for v2
train_dl, val_dl = prepare_trainset_withMask(BATCH_SIZE, NUM_WORKERS, SEED, debug, 
                                    sample_mode='balance', use_sampler=False, IMG_SIZE=IMG_SIZE)#raw,balance, weak_balance

######### Run the training process #########
run_check_net(train_dl, val_dl, multi_gpu=multi_gpu)