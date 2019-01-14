import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from se_resnet import se_resnet50

from loss import FocalLoss
from metrics import macro_f1


class AtlasSEResnet50(nn.Module):

    def __init__(self, debug=False):
        super().__init__()
        
        self.debug = debug
        
        #self.downsize_input299 = nn.AdaptiveAvgPool2d((224, 224))
        self.se_resnet50 = self._load_pretrained_weight()
        self.encoder = nn.Sequential(
            self.se_resnet50.layer0,
            self.se_resnet50.layer1,
            self.se_resnet50.layer2,
            self.se_resnet50.layer3,
            self.se_resnet50.layer4,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # raw: nn.AvgPool2d(7, stride=1)
        self.logit = nn.Sequential(
            #nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512 * 4, 28), #block.expansion=4, num_classes=28
            #nn.Sigmoid()
        )
        
#         self.center = nn.Sequential(
#             ConvBn2d(512, 512, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             ConvBn2d(512, 256, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )

#         self.logit    = nn.Sequential(
#             nn.Conv2d(320, 64, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64,  1, kernel_size=1, padding=0),
#         )
    
    def _load_pretrained_weight(self):
        ## trick: get 4 channels weights from pretrained resnet
        _net = se_resnet50(pretrained='imagenet', input_channel=3)
        state_dict = _net.state_dict().copy()
        layer0_weights = state_dict['layer0.conv1.weight']
        print('raw_weight size: ', layer0_weights.size())
        layer0_weights_new = torch.nn.Parameter(torch.cat((layer0_weights, layer0_weights[:,:1,:,:]),dim=1))
        print('new_weight size: ', layer0_weights_new.size())
        new_state_dict = OrderedDict(('layer0.conv1.weight', layer0_weights_new) if key == 'layer0.conv1.weight' \
                                     else (key, value) for key, value in state_dict.items())
        ## 
        net = se_resnet50(pretrained=None, input_channel=4)
        net.load_state_dict(new_state_dict)
        return net
    
    def forward(self, x):
        #batch_size,C,H,W = x.shape
        
        if self.debug:
            print('input: ', x.size())
#         x = self.downsize_input299(x)
#         if self.debug:
#             print('downsize: ', x.size())

        x = self.encoder(x)
        if self.debug:
            print('se-resnet50_encoder: ', x.size())

        x = self.avgpool(x)
        if self.debug:
            print('avgpool', x.size())
        x = x.view(x.size(0), -1)
        if self.debug:
            print('reshape', x.size())
        logit = self.logit(x)
        if self.debug:
            print('output', logit.size())

        return logit
    
        ##-----------------------------------------------------------------

    def criterion(self, logit, truth):
        """Define the (customized) loss function here."""
        Loss_FUNC = FocalLoss()
        #Loss_FUNC = nn.BCEWithLogitsLoss()#nn.MultiLabelSoftMarginLoss()
        loss = Loss_FUNC(logit, truth)
        return loss

    def metric(self, logit, truth, device='gpu'):
        """Define metrics for evaluation especially for early stoppping."""
        return macro_f1(logit, truth, device=device)

    def set_mode(self, mode):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError


def predict_proba(net, test_dl, device):
    y_pred = None
    net.set_mode('test')
    with torch.no_grad():
        for i, (input_data, truth) in enumerate(test_dl):
            #if i > 10:
            #    break
            input_data, truth = input_data.to(device=device, dtype=torch.float), truth.to(device=device, dtype=torch.float)
            logit = net(input_data).cpu().numpy()
            if y_pred is None:
                y_pred = logit
            else:
                y_pred = np.concatenate([y_pred, logit], axis=0)
    return y_pred.reshape(-1, 28)
