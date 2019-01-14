import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
#import zipfile
from glob import glob
import copy
import pickle
from torch.utils.data import DataLoader, Dataset
from augmentation import do_augmentation
import cv2
from skmultilearn.model_selection import IterativeStratification
import time

import torch


name_label_dict = {
0:  'Nucleoplasm',
1:  'Nuclear membrane',
2:  'Nucleoli',   
3:  'Nucleoli fibrillar center',
4:  'Nuclear speckles',
5:  'Nuclear bodies',
6:  'Endoplasmic reticulum',   
7:  'Golgi apparatus',
8:  'Peroxisomes',
9:  'Endosomes',
10:  'Lysosomes',
11:  'Intermediate filaments',
12:  'Actin filaments',
13:  'Focal adhesion sites',   
14:  'Microtubules',
15:  'Microtubule ends',  
16:  'Cytokinetic bridge',   
17:  'Mitotic spindle',
18:  'Microtubule organizing center',  
19:  'Centrosome',
20:  'Lipid droplets',
21:  'Plasma membrane',   
22:  'Cell junctions', 
23:  'Mitochondria',
24:  'Aggresome',
25:  'Cytosol',
26:  'Cytoplasmic bodies',   
27:  'Rods & rings' }

# some image files are missing when downloading...
BROKEN = \
['22076_615_D10_1',
 '22076_615_D10_1',
 '22076_615_D10_1',
 '23370_182_B12_2',
 '77752_1636_D3_32',
 '77752_1636_D3_32',
 '77752_1636_D3_32',
 '64301_1169_A12_1',
 '64301_1169_A12_1',
 '54431_1179_H7_2',
 '54431_1179_H7_2',
 '54431_1179_H7_2',
 '6104_7_G5_1',
 '57384_1377_F4_3',
 '57384_1377_F4_3',
 '57384_1377_F4_3',
 '41995_540_H5_2',
 '41995_540_H5_2',
 '23874_196_H4_1', #from external_data_part3
 '16992_1606_H3_6', 
 '7979_22_G2_2'
]


def get_external_data(external_data_part3=False):
    """
    update: external_data/, external_data_part3/
    """
    if external_data_part3:
        df = pd.read_csv('data/raw/external_data.csv')
        files = glob('data/raw/external_data/*') + glob('data/raw/external_data_part3/*')
        files = list(set([f.replace('_red.png', '').replace('_green.png', '').replace('_blue.png', '').replace('_yellow.png', '').replace('data/raw/external_data/', '').replace('data/raw/external_data_part3/', '') for f in files]))
        # excluding some missing files (missing channel files)
        files = [f for f in files if f not in BROKEN]
    else:
        df = pd.read_csv('data/raw/external_data.csv')
        files = glob('data/raw/external_data/*')
        files = list(set([f.replace('_red.png', '').replace('_green.png', '').replace('_blue.png', '').replace('_yellow.png', '').replace('data/raw/external_data/', '') for f in files]))
        # excluding some missing files (missing channel files)
        files = [f for f in files if f not in BROKEN]
    return df.loc[df.Id.isin(files), ]

def get_label_data(use_external=True):
    train_label = pd.read_csv('data/raw/train.csv')
    if use_external:
        external_label = get_external_data()
        train_label = pd.concat([train_label, external_label], axis=0)
        train_label.reset_index(drop=True, inplace=True)
    return train_label

def iterative_stratification(X, y, SEED):
    stratifier = IterativeStratification(n_splits=2, order=2, sample_distribution_per_fold=[0.1, 0.9], random_state=SEED)
    train_indexes, test_indexes = next(stratifier.split(X, y))

    X_train, y_train = X[train_indexes, :], y[train_indexes, :]
    X_test, y_test = X[test_indexes, :], y[test_indexes, :]

    return X_train, y_train, X_test, y_test

def train_test_split(SEED, debug=False):
    """
    split dataset on fname using "iterative_stratification"
    #load raw images and targets, convert into dataset and multihot labels
    """
    ##
    #archive = zipfile.ZipFile('data/raw/train.zip', 'r')
    #trainset_fname_list = archive.namelist()
    #trainset_fname_list = list(set([f.split('_')[0] for f in trainset_fname_list]))
    
    ##train test split
    #fname_train, fname_valid = train_test_split(
    #trainset_fname_list,
    #test_size=0.1, random_state=SEED)#stratify=train_df.coverage_class
    
    ##label
    train_label = get_label_data()
    if debug:
        train_label = train_label.sample(n=1000, axis=0, random_state=SEED)
    print('total trainset: ', len(train_label.index))

    ## multi-label stratify train-valid-split
    target_list = []
    for i in train_label.index:
        target = multi_hot(train_label.loc[i, 'Target'])
        target_list.append(target)
    target_arr = np.array(target_list)
    fname_arr = train_label['Id'].values.reshape(-1, 1)
    fname_train, target_train, fname_valid, target_valid = iterative_stratification(fname_arr, target_arr, SEED)
    fname_train, fname_valid = fname_train[:, 0], fname_valid[:, 0]
    return fname_train, fname_valid

#     ##read numpy format data -- x_train & x_valid & y_train & y_valid
#     x_train, x_valid = [], []
#     y_train, y_valid = [], []
#     for i, f in enumerate(tqdm_notebook(fname_train)):
#         l = []
#         for color in ['_red.png', '_green.png', '_blue.png', '_yellow.png']:
#             _f = f + color
#             img_file = archive.open(_f)
#             img_arr = plt.imread(img_file)
#             img_arr = do_resize(img_arr, H=256, W=256)#donwsize from 512->256
#             img_arr = np.expand_dims(img_arr, 0)
#             l.append(img_arr)
#         _x_train = np.concatenate(l, axis=0)
#         _x_train = np.expand_dims(_x_train, 0)
#         x_train.append(_x_train)
#         _y_train_str = train_label.loc[train_label['Id']==f, 'Target'].values[0]
#         _y_train = multi_hot(_y_train_str)
#         y_train.append(_y_train)
#     for i, f in enumerate(tqdm_notebook(fname_valid)):
#         l = []
#         for color in ['_red.png', '_green.png', '_blue.png', '_yellow.png']:
#             _f = f + color
#             img_file = archive.open(_f)
#             img_arr = plt.imread(img_file)
#             img_arr = do_resize(img_arr, H=256, W=256)#donwsize from 512->256
#             img_arr = np.expand_dims(img_arr, 0)
#             l.append(img_arr)
#         _x_valid = np.concatenate(l, axis=0)
#         _x_valid = np.expand_dims(_x_valid, 0)
#         x_valid.append(_x_valid)
#         _y_valid_str = train_label.loc[train_label['Id']==f, 'Target'].values[0]
#         _y_valid = multi_hot(_y_valid_str)
#         y_valid.append(_y_valid)
#     x_train = np.concatenate(x_train, axis=0)
#     x_valid = np.concatenate(x_valid, axis=0)
#     y_train = np.array(y_train)
#     y_valid = np.array(y_valid)
#     return x_train, x_valid, y_train, y_valid, fname_train, fname_valid

def multi_hot(str_target):
    multihot_target = np.zeros(28, dtype=np.int)
    for target in str_target.split(' '):
        target = int(target)
        multihot_target[target] = 1
    return multihot_target


class AtlasDataSet(Dataset):
    def __init__(self, fname, mode='train', augmentation=False, IMG_SIZE=512, flip=False):
        super(AtlasDataSet, self).__init__()
        self.fname = fname#images
        self.mode = mode
        #self.image_path = image_path
        if self.mode == 'train':
            self.label = get_label_data()
        elif self.mode == 'test':
            self.label = np.zeros((len(self.fname), 28))
        self.augmentation = augmentation
        self.IMG_SIZE = IMG_SIZE
        print('IMG_SIZE: ', self.IMG_SIZE)
        self.flip = flip

    def __getitem__(self, idx):
        f = self.fname[idx]
        l = []
        for color in ['_red.png', '_green.png', '_blue.png', '_yellow.png']:
            if self.mode == 'train':
                img_file = glob('data/raw/*/'+f+color)[0]
            elif self.mode == 'test':
                try:
                    img_file = glob('data/raw/test/'+f+color)[0]
                except:
                    print(f, color)
                    raise ValueError('Couldnt find image: ', f)
            img_arr = plt.imread(img_file)
            if self.flip:
                img_arr = np.fliplr(img_arr)#for TTA
            if self.IMG_SIZE != 512:
                img_arr = cv2.resize(img_arr, (self.IMG_SIZE,self.IMG_SIZE))## try larger input image size, training takes too long
            img_arr = np.expand_dims(img_arr, 0)
            l.append(img_arr)
        x = np.concatenate(l, axis=0)
        #print(x.shape): (4, 512, 512)
        ## do augmentation here !!!
        if self.augmentation:
            x = do_augmentation(x)        
        if self.mode == 'train':
            y_str = self.label.loc[self.label['Id']==f, 'Target'].values[0]
            y = multi_hot(y_str)
        elif self.mode == 'test':
            y = self.label[idx]
            #raise ValueError("not implemented for testset")
        return x, y

    def __len__(self):
        return len(self.fname)


class WeakBalancedDataset(object):
    """
    downsample major class: '0', '25'
    
    TODO: upsample minor class? x10?
    """
    def __init__(self, train_label, fname):
        """
        fname: fname_train or fname_valid
        """
        self.train_label = train_label.loc[train_label.Id.isin(fname), ]
        self.fname_list = fname
        #
        #self.major = ['0', '25']
        self.minor = ['8', '9', '10', '15', '27']
        #self.common = [str(c) for c in range(28) if str(c) not in self.major and str(c) not in self.minor]
    
    def make_sample(self, SEED=1234):
        np.random.seed(SEED)
        # downsample major class
        del_list = []
        for i in self.train_label.index:
            label = self.train_label.loc[i, 'Target'].split(' ')
            fname = self.train_label.loc[i, 'Id']
            if '0' not in label and '25' not in label:
                continue
            skip = False
            for mi in self.minor:
                if mi in label:
                    skip = True
                    break
            if skip:
                continue
            del_list.append(fname)
        del_list = np.random.choice(del_list, round(len(del_list)/2))#randomly drop 50% major class samples
        sample_list = [f for f in self.fname_list if f not in del_list]
        # upsample minor class
        upsample_list = []
        for i in self.train_label.index:
            label = self.train_label.loc[i, 'Target'].split(' ')
            fname = self.train_label.loc[i, 'Id']
            for i in self.minor:
                if i in label:
                    upsample_list.append(fname)
                    break
        sample_list.extend(upsample_list*10)
        return sample_list
        

class BalancedDataset(object):
    """
    make a dataset that each class has equal number of samples
    
    each batch e.g.(batch size=32) has samples like (by class): [0, 1, ..., 27, rand0, rand1, rand2, rand3] 
    
    example:
    balanced_dataset = BalancedDataset(train_label, fname_train)
    train_sample = balanced_dataset.make_sample(n_iter=1000, BATCH_SIZE=32, SEED=1234)
    balanced_dataset = BalancedDataset(train_label, fname_valid)
    valid_sample = balanced_dataset.make_sample(n_iter=100, BATCH_SIZE=32, SEED=1234)
    """
    def __init__(self, train_label, fname):
        """
        fname: fname_train or fname_valid
        """
        self.train_label = train_label.loc[train_label.Id.isin(fname), ]
        self.fname_list = fname
        self._make_target_arr()
    
    def _make_target_arr(self):
        target_list = []
        for i in self.train_label.index:
            target = multi_hot(self.train_label.loc[i, 'Target'])
            target_list.append(target)
        self.target_arr = np.array(target_list)

    def make_sample(self, n_iter=1000, BATCH_SIZE=32, SEED=1234):
        """
        n_iter: 1000, so 1000x28~30000
        """
        np.random.seed(SEED)
        sample_list = []
        for iteration in range(n_iter):
            ##
            sample = []
            # here, each class has one sample point in this batch
            for i in range(28):
                _fname_list = self.train_label.loc[self.target_arr[:, i]==1, 'Id'].tolist()
                if _fname_list == []:
                    _sample = np.random.choice(self.fname_list, 1)[0]
                else:
                    _sample = np.random.choice(_fname_list, 1)[0]
                sample.append(_sample)
            # here, the rest samples have random classes
            for i in range(BATCH_SIZE-28):
                _sample = np.random.choice(self.fname_list, 1)[0]
                sample.append(_sample)
            #sample_list.append(sample)
            sample_list.extend(sample)
        return sample_list

def prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, debug=False, sample_mode='balance', use_sampler=False, IMG_SIZE=512):
    """
    load saved dataset, and use torch DataLoader
    """
    # read numpy format data
    #with open('data/processed/trainset_%d.pkl'%SEED, 'rb') as f:
    #    x_train, x_valid, y_train, y_valid, fname_train, fname_valid = pickle.load(f)
    #y_train = y_train.astype(np.int)
    #y_valid = y_valid.astype(np.int)
    #if debug:
    #    x_train, y_train = x_train[:500], y_train[:500]
    #    x_valid, y_valid = x_valid[:100], y_valid[:100]
    fname_train, fname_valid = train_test_split(SEED, debug)
    #if debug:
    #    fname_train, fname_valid = fname_train[:500], fname_valid[:100]
    print('Count of trainset: ', len(fname_train))
    print('Count of validset: ', len(fname_valid))

    ##label
    train_label = get_label_data()
    if sample_mode=='balance':
        balanced_dataset = BalancedDataset(train_label, fname_train)
        train_sample = balanced_dataset.make_sample(n_iter=1500, BATCH_SIZE=32, SEED=SEED)#1500
    elif sample_mode=='weak_balance':
        weak_balanced_dataset = WeakBalancedDataset(train_label, fname_train)
        train_sample = weak_balanced_dataset.make_sample(SEED=SEED)
    elif sample_mode=='raw':
        train_sample = fname_train
    else:
        raise ValueError('sample_mode not identified')
    
    #balanced_dataset = BalancedDataset(train_label, fname_valid)
    if debug:
        #valid_sample = balanced_dataset.make_sample(n_iter=10, BATCH_SIZE=32, SEED=SEED)
        train_sample = np.random.choice(train_sample, 900, replace=True).tolist()
        valid_sample = np.random.choice(fname_valid, 100, replace=True).tolist()
    else:
        valid_sample = fname_valid
        #valid_sample = balanced_dataset.make_sample(n_iter=100, BATCH_SIZE=32, SEED=SEED)
    
    print('Count of trainset (for training): ', len(train_sample))
    print('Count of validset (for training): ', len(valid_sample))
    # make pytorch.data.Dataset
    train_ds = AtlasDataSet(train_sample, mode='train', augmentation=True, IMG_SIZE=IMG_SIZE)#False
    val_ds = AtlasDataSet(valid_sample, mode='train', augmentation=False, IMG_SIZE=IMG_SIZE)
    
    #class_weights for pytorch sampler
    if use_sampler:
        class_weights = make_class_weights(train_sample, train_label, modify=False)
        #print(class_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=class_weights, 
                                                                 num_samples=len(class_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,#for sampler/BalancedDataset, this must be False because it requires the order of samples points
        sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        #sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    
    return train_dl, val_dl

def prepare_trainset_withMask(BATCH_SIZE, NUM_WORKERS, SEED, debug=False, sample_mode='balance', use_sampler=False, IMG_SIZE=512):
    """
    load saved dataset, and use torch DataLoader
    """
    # read numpy format data
    #with open('data/processed/trainset_%d.pkl'%SEED, 'rb') as f:
    #    x_train, x_valid, y_train, y_valid, fname_train, fname_valid = pickle.load(f)
    #y_train = y_train.astype(np.int)
    #y_valid = y_valid.astype(np.int)
    #if debug:
    #    x_train, y_train = x_train[:500], y_train[:500]
    #    x_valid, y_valid = x_valid[:100], y_valid[:100]
    fname_train, fname_valid = train_test_split(SEED, debug)
    #if debug:
    #    fname_train, fname_valid = fname_train[:500], fname_valid[:100]
    print('Count of trainset: ', len(fname_train))
    print('Count of validset: ', len(fname_valid))

    ##label
    train_label = get_label_data()
    if sample_mode=='balance':
        balanced_dataset = BalancedDataset(train_label, fname_train)
        train_sample = balanced_dataset.make_sample(n_iter=1500, BATCH_SIZE=32, SEED=SEED)#1500
    elif sample_mode=='weak_balance':
        weak_balanced_dataset = WeakBalancedDataset(train_label, fname_train)
        train_sample = weak_balanced_dataset.make_sample(SEED=SEED)
    elif sample_mode=='raw':
        train_sample = fname_train
    else:
        raise ValueError('sample_mode not identified')
    
    #balanced_dataset = BalancedDataset(train_label, fname_valid)
    if debug:
        #valid_sample = balanced_dataset.make_sample(n_iter=10, BATCH_SIZE=32, SEED=SEED)
        train_sample = np.random.choice(train_sample, 900, replace=True).tolist()
        valid_sample = np.random.choice(fname_valid, 100, replace=True).tolist()
    else:
        valid_sample = fname_valid
        #valid_sample = balanced_dataset.make_sample(n_iter=100, BATCH_SIZE=32, SEED=SEED)
    
    print('Count of trainset (for training): ', len(train_sample))
    print('Count of validset (for training): ', len(valid_sample))
    # make pytorch.data.Dataset
    train_ds = AtlasDataSetWithMask(train_sample, mode='train', augmentation=True, IMG_SIZE=IMG_SIZE)#False
    val_ds = AtlasDataSetWithMask(valid_sample, mode='train', augmentation=False, IMG_SIZE=IMG_SIZE)
    
    #class_weights for pytorch sampler
    if use_sampler:
        class_weights = make_class_weights(train_sample, train_label, modify=False)
        #print(class_weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=class_weights, 
                                                                 num_samples=len(class_weights), replacement=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True
    
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,#for sampler/BalancedDataset, this must be False because it requires the order of samples points
        sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        #sampler=sampler,
        num_workers=NUM_WORKERS,
        drop_last=True
    )
    
    return train_dl, val_dl

class AtlasDataSetWithMask(Dataset):
    def __init__(self, fname, mode='train', augmentation=False, IMG_SIZE=512, flip=False):
        super(AtlasDataSetWithMask, self).__init__()
        self.fname = fname#images
        self.mode = mode
        #self.image_path = image_path
        if self.mode == 'train':
            self.label = get_label_data()
        elif self.mode == 'test':
            self.label = np.zeros((len(self.fname), 28))
        self.augmentation = augmentation
        self.IMG_SIZE = IMG_SIZE
        print('IMG_SIZE: ', self.IMG_SIZE)
        self.flip = flip

    def __getitem__(self, idx):
        f = self.fname[idx]
        l = []
        for color in ['_red.png', '_green.png', '_blue.png']:#'_yellow.png'
            if self.mode == 'train':
                img_file = glob('data/raw/*/'+f+color)[0]
            elif self.mode == 'test':
                try:
                    img_file = glob('data/raw/test/'+f+color)[0]
                except:
                    print(f, color)
                    raise ValueError('Couldnt find image: ', f)
            img_arr = plt.imread(img_file)
            if self.flip:
                img_arr = np.fliplr(img_arr)#for TTA
            if self.IMG_SIZE != 512:
                img_arr = cv2.resize(img_arr, (self.IMG_SIZE,self.IMG_SIZE))## try larger input image size, training takes too long
            img_arr = np.expand_dims(img_arr, 0)
            l.append(img_arr)
        x = np.concatenate(l, axis=0)
        mask = do_segment_mask(x[1, :, :], window_size=16) #on green channel only
        
        ## do augmentation here !!!
        if self.augmentation:
            x, mask = do_augmentation(x, mask)    #image:(4, 512, 512), mask:(1, 512, 512)
        else:
            mask = mask.reshape(1, 512, 512)
        if self.mode == 'train':
            y_str = self.label.loc[self.label['Id']==f, 'Target'].values[0]
            y = multi_hot(y_str)
        elif self.mode == 'test':
            y = self.label[idx]
            #raise ValueError("not implemented for testset")
        #print(x.shape, mask.shape, y.shape)
        return np.concatenate((x, mask), axis=0), y #x, mask, y

    def __len__(self):
        return len(self.fname)

def do_segment_mask(img, window_size=16):
    """create cell masks, return numpy.array with 0-1
    """
    #step1: binary thresholding, but lots of isolated noises
    #mask = (img > np.percentile(img, 50)).astype(np.float32)
    #step2: smoothing(or local averaging), to denoise
    kernel = np.ones((window_size, window_size),np.float32) / window_size**2 #5x5 windows
    smoothed_mask = cv2.filter2D(img, ddepth=-1, kernel=kernel)#
    #step3: binary threshold again
    final_mask = (smoothed_mask > np.percentile(smoothed_mask, 75)).astype(np.float32)
    return final_mask

def make_class_weights(fname, label, modify=False):
    """
    The class_weights parameter in pytorch sampler is a array for assigning weighter for each sample point in order.
    """
    torch.multiprocessing.set_sharing_strategy('file_system')
    #based on official dataset
    class_weights_arr = np.array([1.0, 3.01, 1.95, 2.79, 2.61, 2.31, 3.23, 2.2, 6.17, 
                                  6.34, 6.81, 3.15, 3.61, 3.86, 3.17, 7.1, 3.87, 4.8, 
                                  3.34, 2.84, 4.99, 1.91, 3.46, 2.15, 4.37, 1.13, 4.35, 7.74])
    #plus external dataset
    #class_weights_arr = np.array([1.07, 3.07, 1.79, 2.99, 2.56, 2.42, 2.88, 1.94, 5.75, 
    #                              5.8, 5.88, 3.4, 3.39, 3.82, 3.2, 7.04, 3.94, 4.99, 
    #                              3.55, 2.89, 5.04, 1.55, 3.19, 1.84, 5.04, 1.08, 4.54, 6.24])
    
    # if this sample is minor, remove its 0/25 class (over-sample minor class)
    #minor = ['8', '9', '10', '15', '20', '27']

    class_weights = []

    print('calculate label class_weights')
    t0 = time.time()
    for f in fname:#tqdm_notebook
        target = label.loc[label.Id==f, 'Target'].values[0]
        target_split = target.split(' ')
        target = multi_hot(target)
        #
        if modify:
            if len(target_split)>=2 and (('0' in target_split and '25' not in target_split) or ('0' not in target_split and '25' in target_split)):
                target *= np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 0, 1, 1])
            elif len(target_split)>2 and ('0' in target_split and '25' in target_split):
                target *= np.array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                   1, 1, 1, 0, 1, 1])
        #
        a = (class_weights_arr * target)
        weight = a[a.nonzero()].mean()#.max()
        class_weights.append(weight)
    print((time.time()-t0)//60, ' min')
    return class_weights



if __name__ == "__main__":
    SEED = 1234
    # BATCH_SIZE = 16
    # NUM_WORKERS = 20
    debug = False#True
    x_train, x_valid, y_train, y_valid, fname_train, fname_valid = make_trainset(SEED, debug)
    
    ## save trainset
    with open('data/processed/trainset_%d.pkl'%SEED, 'wb') as f:#trainset_aug_%d.pkl
        pickle.dump([x_train, x_valid, y_train, y_valid, fname_train, fname_valid], f)
    print(x_train.shape, y_train.shape)
