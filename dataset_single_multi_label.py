import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
#from sklearn.model_selection import train_test_split
from tqdm import tqdm_notebook
#import zipfile
from glob import glob
import pickle
from torch.utils.data import DataLoader, Dataset
from augmentation import do_augmentation
import cv2
#from skmultilearn.model_selection import IterativeStratification
from sklearn.model_selection import train_test_split as train_test_split_sklearn
import time

import torch


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

def train_test_split(SEED, debug=False):
    """single/multi label nearly 50/50, so random split is OK
    """
    ##label
    train_label = get_label_data()
    if debug:
        train_label = train_label.sample(n=1000, axis=0, random_state=SEED)
    print('total trainset: ', len(train_label.index))

    ## stratify train-valid-split
    target_list = []
    for i in train_label.index:
        target = onehot(train_label.loc[i, 'Target'])
        target_list.append(target)
    target_arr = np.array(target_list)
    fname_arr = train_label['Id'].values.reshape(-1, 1)
    fname_train, fname_valid = train_test_split_sklearn(fname_arr, test_size=0.1, random_state=SEED, stratify=target_arr)
    fname_train, fname_valid = fname_train[:, 0], fname_valid[:, 0]
    return fname_train, fname_valid

def onehot(str_target):
    l = str_target.split(' ')
    if len(l)==1:
        onehot_target = np.array([0, 1])#single label -- positive
    elif len(l)>1:
        onehot_target = np.array([1, 0])#multi label -- negtive
    return onehot_target


class AtlasDataSet(Dataset):
    def __init__(self, fname, mode='train', augmentation=False, IMG_SIZE=512):
        super(AtlasDataSet, self).__init__()
        self.fname = fname#images
        self.mode = mode
        #self.image_path = image_path
        if self.mode == 'train':
            self.label = get_label_data()
        elif self.mode == 'test':
            self.label = np.zeros((len(self.fname), 2))
        self.augmentation = augmentation
        self.IMG_SIZE = IMG_SIZE
        print('IMG_SIZE: ', self.IMG_SIZE)

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
            if self.IMG_SIZE != 512:
                img_arr = cv2.resize(img_arr, (self.IMG_SIZE,self.IMG_SIZE))## try larger input image size, training takes too long
            img_arr = np.expand_dims(img_arr, 0)
            l.append(img_arr)
        x = np.concatenate(l, axis=0)
        ## do augmentation here !!!
        if self.augmentation:
            x = do_augmentation(x)        
        if self.mode == 'train':
            y_str = self.label.loc[self.label['Id']==f, 'Target'].values[0]
            y = onehot(y_str)
        elif self.mode == 'test':
            y = self.label[idx]
            #raise ValueError("not implemented for testset")
        return x, y

    def __len__(self):
        return len(self.fname)

def prepare_trainset(BATCH_SIZE, NUM_WORKERS, SEED, debug=False, IMG_SIZE=512):
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
    train_sample = fname_train

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
#     if use_sampler:
#         class_weights = make_class_weights(train_sample, train_label, modify=False)
#         #print(class_weights)
#         sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=class_weights, 
#                                                                  num_samples=len(class_weights), replacement=True)
#         shuffle = False
#     else:
#         sampler = None
#         shuffle = True
    
    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,#for sampler/BalancedDataset, this must be False because it requires the order of samples points
        #sampler=None,
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
