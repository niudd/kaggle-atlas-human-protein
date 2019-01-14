import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import f1_score as f1_score_sklearn
import numpy as np


def macro_f1(logit, truth, threshold=0.5, device='gpu'):#_macro_f1
    logit = torch.sigmoid(logit)
    if device=='cpu':
        pred = (logit.detach().numpy() > threshold).astype(np.int)
        truth = truth.detach().numpy().astype(np.int)
    elif device=='gpu':
        pred = (logit.cpu().detach().numpy() > threshold).astype(np.int)
        truth = truth.cpu().detach().numpy().astype(np.int)
    return f1_score_sklearn(truth, pred, average='macro')


# def macro_f1(logit, truth, device='gpu'):
#     if device=='gpu':#for train
#         return _macro_f1(logit, truth, threshold=0.0, device='gpu')
#     elif device=='cpu':#for valid
#         thresholds_candidates = np.linspace(-2, 2, 200)
#         thresholds = np.array([0.5]*28)
#         f1_score_best_list = []
#         for c in range(28):
#             f1_scores = []
#             for i in thresholds_candidates:
#                 thresholds[c] = i

#                 f1_score = _macro_f1(logit, truth, threshold=thresholds, device='cpu')
#                 f1_scores.append(f1_score)

#             # best threshold for this class
#             threshold_best_index = np.argmax(f1_scores)
#             f1_score_best = f1_scores[threshold_best_index]
#             threshold_best = thresholds_candidates[threshold_best_index]
#             ## add f1_score_best to history, and set threshold best for this class
#             f1_score_best_list.append(f1_score_best)
#             thresholds[c] = threshold_best
#         return f1_score_best_list[-1]

def macro_f1_numpy(logit, truth, threshold=0.0):#threshold=0.5
    pred = (logit > threshold).astype(np.int)
    truth = truth.astype(np.int)
    return f1_score_sklearn(truth, pred, average='macro')

