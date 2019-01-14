import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from inception import inception_v3

from loss import FocalLoss
from metrics import macro_f1


class AtlasInceptionV3(nn.Module):

    def __init__(self, debug=False, num_classes=28, aux_logits=True, transform_input=False):
        super().__init__()
        
        self.debug = debug
        self.num_classes = num_classes
        self.aux_logits = aux_logits
        self.transform_input = transform_input
        
        self.downsize_input299 = nn.AdaptiveAvgPool2d((299, 299))
        self.inception = self._load_pretrained_weight(channels=4, num_classes=num_classes, 
                                                      aux_logits=aux_logits, transform_input=transform_input)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))#F.avg_pool2d(x, kernel_size=8)
        self.logit = nn.Sequential(
            #nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(2048, 28), #block.expansion=1, num_classes=28
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

    def _load_pretrained_weight(self, channels=4, num_classes=28, aux_logits=True, transform_input=False):
        ## trick: get 4 channels weights from pretrained resnet
        _net = torchvision.models.inception_v3(pretrained=True)
        state_dict = _net.state_dict().copy()
        layer0_weights = state_dict['Conv2d_1a_3x3.conv.weight']
        print('raw_weight size: ', layer0_weights.size())
        layer0_weights_new = torch.nn.Parameter(torch.cat((layer0_weights, layer0_weights[:,:1,:,:]),dim=1))
        print('new_weight size: ', layer0_weights_new.size())
        new_state_dict = []
        for key, value in state_dict.items():
            if not aux_logits and 'AuxLogits' in key:
                continue
            
            if key == 'Conv2d_1a_3x3.conv.weight':
                new_state_dict.append(('Conv2d_1a_3x3.conv.weight', layer0_weights_new))
            elif key in ['fc.weight', 'fc.bias']:
                continue
            elif key == 'AuxLogits.fc.weight':
                new_state_dict.append(('AuxLogits.fc.weight',
                                       nn.Parameter(nn.init.xavier_uniform_(torch.empty((28, 768), requires_grad=True)))))
            elif key == 'AuxLogits.fc.bias':
                new_state_dict.append(('AuxLogits.fc.bias',
                                       nn.Parameter(torch.zeros(28, requires_grad=True))))
            else:
                new_state_dict.append((key, value))
        new_state_dict = OrderedDict((k,v) for k,v in new_state_dict)
        ## 
        net = inception_v3(pretrained=False, num_classes=num_classes, 
                           aux_logits=aux_logits, transform_input=transform_input)
        net.load_state_dict(new_state_dict)
        return net
    
    def forward(self, x):
        #batch_size,C,H,W = x.shape
        
        if self.debug:
            print('input: ', x.size())
        x = self.downsize_input299(x)
        if self.debug:
            print('downsize: ', x.size())

        if self.training and self.aux_logits:
            x, aux = self.inception(x)
            if self.debug:
                print('inception: ', x.size(), aux.size())
        else:
            x = self.inception(x)
            if self.debug:
                print('inception: ', x.size())
        
        x = self.avgpool(x)
        if self.debug:
            print('avgpool', x.size())
        x = x.view(x.size(0), -1)
        if self.debug:
            print('reshape', x.size())
        logit = self.logit(x)
        if self.debug:
            print('output', logit.size())

        #f = self.center(e5)
        #if self.debug:
        #    print('center',f.size())

        #f = F.dropout(f, p=0.40)#training=self.training
        #logit = self.logit(f)
        #if self.debug:
        #    print('logit', logit.size())
        if self.training and self.aux_logits:
            return logit, aux
        return logit
    
        ##-----------------------------------------------------------------

    def criterion(self, logit, truth):
        """Define the (customized) loss function here."""
        Loss_FUNC = FocalLoss()
        #Loss_FUNC = nn.BCEWithLogitsLoss()#nn.MultiLabelSoftMarginLoss()
        if self.training and self.aux_logits:
            logit, aux = logit[0], logit[1]
            return Loss_FUNC(logit, truth) + 0.3 * Loss_FUNC(aux, truth)#according to paper, aux_loss_weight=0.3
        loss = Loss_FUNC(logit, truth)
        return loss

    def metric(self, logit, truth):
        """Define metrics for evaluation especially for early stoppping."""
        if self.training and self.aux_logits:
            logit = logit[0]
            return macro_f1(logit, truth)
        return macro_f1(logit, truth)

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
