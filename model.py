import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from collections import OrderedDict
from resnet import resnet34, resnet18

from loss import FocalLoss, f1_loss
from metrics import macro_f1


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        #self.dropout = nn.Dropout2d(p=0.1, inplace=False)
        self.bn = nn.BatchNorm2d(out_channels)
        #self.bn = SynchronizedBatchNorm2d(out_channels)

    def forward(self, z):
        x = self.conv(z)
        #x = self.dropout(x)
        x = self.bn(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 =  ConvBn2d(in_channels,  channels, kernel_size=3, padding=1)
        self.conv2 =  ConvBn2d(channels, out_channels, kernel_size=3, padding=1)
        self.spa_cha_gate = SCSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)#False
        if e is not None:
            x = torch.cat([x, e], 1)
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = self.spa_cha_gate(x)
        return x

class SCSE(nn.Module):
    def __init__(self, in_ch):
        super(SCSE, self).__init__()
        self.spatial_gate = SpatialGate2d(in_ch, 16)#16
        self.channel_gate = ChannelGate2d(in_ch)
    
    def forward(self, x):
        g1 = self.spatial_gate(x)
        g2 = self.channel_gate(x)
        x = g1 + g2 #x = g1*x + g2*x
        return x

class SpatialGate2d(nn.Module):
    def __init__(self, in_ch, r=16):
        super(SpatialGate2d, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = x.view(*(x.shape[:-2]),-1).mean(-1)
        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class ChannelGate2d(nn.Module):
    def __init__(self, in_ch):
        super(ChannelGate2d, self).__init__()

        self.conv = nn.Conv2d(in_ch, 1, kernel_size=1, stride=1)

    def forward(self, x):
        input_x = x

        x = self.conv(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class AtlasResNet34(nn.Module):

    def __init__(self, pretrained=True, debug=False, num_classes=28):
        super().__init__()
        #self.resnet = resnet34(pretrained=False) #torchvision.models.resnet34(pretrained=pretrained)
        self.resnet = self._load_pretrained_weight(channels=4)
        self.debug = debug

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool,
        )# 64
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCSE(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCSE(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCSE(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCSE(512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit = nn.Sequential(
            #nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512 * 1, num_classes), #block.expansion=1, num_classes=28
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

    def _load_pretrained_weight(self, channels=4):
        ## trick: get 4 channels weights from pretrained resnet
        _net = torchvision.models.resnet34(pretrained=True)
        state_dict = _net.state_dict().copy()
        layer0_weights = state_dict['conv1.weight']
        print('raw_weight size: ', layer0_weights.size())
        layer0_weights_new = torch.nn.Parameter(torch.cat((layer0_weights, layer0_weights[:,:1,:,:]),dim=1))
        print('new_weight size: ', layer0_weights_new.size())
        new_state_dict = OrderedDict(('conv1.weight', layer0_weights_new) if key == 'conv1.weight' \
                                     else (key, value) for key, value in state_dict.items())
        ## 
        net = resnet34(pretrained=False)
        net.load_state_dict(new_state_dict)
        return net
    
    def forward(self, x):
        #batch_size,C,H,W = x.shape

        #x = add_depth_channels(x)
        
        if self.debug:
            print('input: ', x.size())

        x = self.conv1(x)
        if self.debug:
            print('e1',x.size())
        e2 = self.encoder2(x)
        if self.debug:
            print('e2',e2.size())
        e3 = self.encoder3(e2)
        if self.debug:
            print('e3',e3.size())
        e4 = self.encoder4(e3)
        if self.debug:
            print('e4',e4.size())
        e5 = self.encoder5(e4)
        if self.debug:
            print('e5',e5.size())
        
        f = self.avgpool(e5)
        if self.debug:
            print('avgpool',f.size())
        f = f.view(f.size(0), -1)
        if self.debug:
            print('reshape',f.size())
        logit = self.logit(f)
        if self.debug:
            print('output',logit.size())

        #f = self.center(e5)
        #if self.debug:
        #    print('center',f.size())

        #f = F.dropout(f, p=0.40)#training=self.training
        #logit = self.logit(f)
        #if self.debug:
        #    print('logit', logit.size())
        return logit
    
        ##-----------------------------------------------------------------

    def criterion(self, logit, truth):
        """Define the (customized) loss function here."""
        #Loss_FUNC = FocalLoss()
        #Loss_FUNC = nn.BCEWithLogitsLoss()#nn.MultiLabelSoftMarginLoss()
        #loss = Loss_FUNC(logit, truth)
        loss = f1_loss(logit, truth)
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


class AtlasResNet18(nn.Module):

    def __init__(self, pretrained=True, debug=False):
        super().__init__()
        self.resnet = self._load_pretrained_weight(channels=4)
        self.debug = debug

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            #self.resnet.maxpool,
        )# 64
        self.encoder2 = nn.Sequential(self.resnet.layer1, SCSE(64))
        self.encoder3 = nn.Sequential(self.resnet.layer2, SCSE(128))
        self.encoder4 = nn.Sequential(self.resnet.layer3, SCSE(256))
        self.encoder5 = nn.Sequential(self.resnet.layer4, SCSE(512))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.logit = nn.Sequential(
            #nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512 * 1, 28), #block.expansion=1, num_classes=28
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

    def _load_pretrained_weight(self, channels=4):
        ## trick: get 4 channels weights from pretrained resnet
        _net = torchvision.models.resnet18(pretrained=True)
        state_dict = _net.state_dict().copy()
        layer0_weights = state_dict['conv1.weight']
        print('raw_weight size: ', layer0_weights.size())
        layer0_weights_new = torch.nn.Parameter(torch.cat((layer0_weights, layer0_weights[:,:1,:,:]),dim=1))
        print('new_weight size: ', layer0_weights_new.size())
        new_state_dict = OrderedDict(('conv1.weight', layer0_weights_new) if key == 'conv1.weight' \
                                     else (key, value) for key, value in state_dict.items())
        ## 
        net = resnet18(pretrained=False)
        net.load_state_dict(new_state_dict)
        return net
    
    def forward(self, x):
        #batch_size,C,H,W = x.shape

        #x = add_depth_channels(x)
        
        if self.debug:
            print('input: ', x.size())

        x = self.conv1(x)
        if self.debug:
            print('e1',x.size())
        e2 = self.encoder2(x)
        if self.debug:
            print('e2',e2.size())
        e3 = self.encoder3(e2)
        if self.debug:
            print('e3',e3.size())
        e4 = self.encoder4(e3)
        if self.debug:
            print('e4',e4.size())
        e5 = self.encoder5(e4)
        if self.debug:
            print('e5',e5.size())
        
        f = self.avgpool(e5)
        if self.debug:
            print('avgpool',f.size())
        f = f.view(f.size(0), -1)
        if self.debug:
            print('reshape',f.size())
        logit = self.logit(f)
        if self.debug:
            print('output',logit.size())

        #f = self.center(e5)
        #if self.debug:
        #    print('center',f.size())

        #f = F.dropout(f, p=0.40)#training=self.training
        #logit = self.logit(f)
        #if self.debug:
        #    print('logit', logit.size())
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


def predict_proba(net, test_dl, device, multi_gpu=False):
    y_pred = None
    if multi_gpu:
        net.module.set_mode('test')
    else:
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

