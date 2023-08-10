import numpy as np

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics.cluster import contingency_matrix

from .utils import jread


class AnnotationEvaluation(object):
    """Evaluate face annotation"""

    def __init__(self, true_annotations_path, pred_annotations_path):

        self.true_annotations_path = true_annotations_path
        self.pred_annotations_path = pred_annotations_path


    def evaluate(self, verbose=True):

        if verbose:
            print('Reading data ...')
        y_true, y_pred = self._readdata()

        if verbose:
            print('Evaluating ...')
        scores = self.calculate_scores(y_true, y_pred)

        if verbose:
            print('Annotations Report:')
            for k, v in scores.items():
                print('\t{}: {}'.format(k, v))

        return scores


    def calculate_scores(self, y_true, y_pred):

        scores = {
            'Samples': len(y_pred),
            'True Clusters': np.unique(y_true).shape[0],
            'Pred Clusters': np.unique(y_pred).shape[0],
            'Accuracy': np.mean(y_true == y_pred),
            'Precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'Adjusted Rand': adjusted_rand_score(y_true, y_pred),
            'NMI': normalized_mutual_info_score(y_true, y_pred),
            'Purity': self.purity_score(y_true, y_pred),
            'Pairwise': self.pairwise_score(y_true, y_pred),
            'BCubed': self.bcubed_score(y_true, y_pred),
        }

        return scores


    def _readdata(self):

        true_annotations = jread(self.true_annotations_path)
        pred_annotations = jread(self.pred_annotations_path)

        y_true, y_pred = self._aligndata(true_annotations, pred_annotations)

        assert len(y_true) == len(y_pred)

        return y_true, y_pred


    def _aligndata(self, true_annotations, pred_annotations):

        true_annotations = true_annotations['labels']
        pred_annotations = pred_annotations['labels']

        y_true, y_pred = [], []
        for name in pred_annotations.keys():
            try:
                l1 = int(true_annotations[name]['label'])
                l2 = int(pred_annotations[name]['label'])
                y_true.append(l1)
                y_pred.append(l2)
            except:
                continue
        
        return np.array(y_true), np.array(y_pred)


    def purity_score(self, labels_true, labels_pred):

        # Source: https://stackoverflow.com/questions/34047540/python-clustering-purity-metric

        conmat = contingency_matrix(labels_true, labels_pred)
        purity = np.sum(np.amax(conmat, axis=0)) / np.sum(conmat)

        return  purity

    # Source: https://github.com/yl-1993/learn-to-cluster/blob/master/evaluation/metrics.py

    def pairwise_score(self, gt_labels, pred_labels, sparse=True):

        return self._fowlkes_mallows_score(gt_labels, pred_labels, sparse)


    def bcubed_score(self, gt_labels, pred_labels):

        gt_lb2idxs = self._get_lb2idxs(gt_labels)
        pred_lb2idxs = self._get_lb2idxs(pred_labels)

        num_lbs = len(gt_lb2idxs)
        pre = np.zeros(num_lbs)
        rec = np.zeros(num_lbs)
        gt_num = np.zeros(num_lbs)

        for i, gt_idxs in enumerate(gt_lb2idxs.values()):
            all_pred_lbs = np.unique(pred_labels[gt_idxs])
            gt_num[i] = len(gt_idxs)
            for pred_lb in all_pred_lbs:
                pred_idxs = pred_lb2idxs[pred_lb]
                n = 1. * np.intersect1d(gt_idxs, pred_idxs).size
                pre[i] += n**2 / len(pred_idxs)
                rec[i] += n**2 / gt_num[i]

        gt_num = gt_num.sum()
        avg_pre = pre.sum() / gt_num
        avg_rec = rec.sum() / gt_num
        fscore = self._compute_fscore(avg_pre, avg_rec)

        return avg_pre, avg_rec, fscore


    def _fowlkes_mallows_score(self, gt_labels, pred_labels, sparse=True):

        n_samples, = gt_labels.shape

        c = contingency_matrix(gt_labels, pred_labels, sparse=sparse)
        tk = np.dot(c.data, c.data) - n_samples
        pk = np.sum(np.asarray(c.sum(axis=0)).ravel()**2) - n_samples
        qk = np.sum(np.asarray(c.sum(axis=1)).ravel()**2) - n_samples

        avg_pre = tk / pk
        avg_rec = tk / qk
        fscore = self._compute_fscore(avg_pre, avg_rec)

        return avg_pre, avg_rec, fscore

    
    def _compute_fscore(self, pre, rec):
        return 2. * pre * rec / (pre + rec)


    def _get_lb2idxs(self, labels):

        lb2idxs = {}
        for idx, lb in enumerate(labels):
            if lb not in lb2idxs:
                lb2idxs[lb] = []
            lb2idxs[lb].append(idx)
        return lb2idxs
