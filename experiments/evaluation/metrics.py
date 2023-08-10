"""
Source: https://github.com/grigorisg9gr/polynomial_nets/blob/master/face_recognition/eval/lfw.py
"""

import numpy as np

from scipy import interpolate
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_distances


def evaluate(embeddings1, embeddings2, actual_issame, nrof_folds=10, far_target=1e-3):

    thresholds = np.arange(0,1, 0.01)
    tpr, fpr, accuracy = _calculate_roc(
        thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=nrof_folds)
    
    thresholds = np.arange(0, 1, 0.001)
    tar, tar_std, far = _calculate_tar(
        thresholds, embeddings1, embeddings2, actual_issame, far_target=far_target, nrof_folds=nrof_folds)

    return tpr, fpr, accuracy, tar, tar_std, far


def _calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10):

    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    dist = np.diag(cosine_distances(embeddings1, embeddings2))

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = _calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = _calculate_accuracy(threshold, dist[test_set], actual_issame[test_set])
        _, _, accuracy[fold_idx] = _calculate_accuracy(thresholds[best_threshold_index], dist[test_set], actual_issame[test_set])
          
    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)

    return tpr, fpr, accuracy


def _calculate_accuracy(threshold, dist, actual_issame):

    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))
  
    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size

    return tpr, fpr, acc


def _calculate_tar(thresholds, embeddings1, embeddings2, actual_issame, far_target, nrof_folds=10):

    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])

    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)
    
    tar = np.zeros(nrof_folds)
    far = np.zeros(nrof_folds)
    
    dist = np.diag(cosine_distances(embeddings1, embeddings2))
    indices = np.arange(nrof_pairs)
    
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
      
        # Find the threshold that gives FAR = far_target
        far_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            _, far_train[threshold_idx] = _calculate_tar_far(threshold, dist[train_set], actual_issame[train_set])
        if np.max(far_train)>=far_target:
            f = interpolate.interp1d(far_train, thresholds, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0
    
        tar[fold_idx], far[fold_idx] = _calculate_tar_far(threshold, dist[test_set], actual_issame[test_set])
  
    tar_mean = np.mean(tar)
    tar_std = np.std(tar)
    far_mean = np.mean(far)

    return tar_mean, tar_std, far_mean


def _calculate_tar_far(threshold, dist, actual_issame):

    predict_issame = np.less(dist, threshold)
    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    n_same = np.sum(actual_issame)
    n_diff = np.sum(np.logical_not(actual_issame))
    tar = float(true_accept) / float(n_same)
    far = float(false_accept) / float(n_diff)

    return tar, far
