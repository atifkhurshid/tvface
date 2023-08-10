import time
import fastcluster

import numpy as np
import scipy.cluster.hierarchy as hac


class BatchedHierarchicalClustering(object):
    
    def __init__(self, metric, linkage, threshold, batchsize):

        self.metric = metric
        self.linkage_method = linkage
        self.threshold = threshold
        self.batchsize = batchsize


    def cluster(self, embeddings):

        embeddings = np.array(embeddings)

        s = time.time()

        local_labels = self.hac(embeddings)

        elapsed = time.time() - s
        print('Batch HAC @ {:.3f}: Clusters {} | Time {:.3f}'.format(
            self.threshold, len(set(local_labels)), elapsed))

        s = time.time()

        mean_embeddings = np.array([
            embeddings[local_labels == y].mean(axis=0)
            for y in sorted(set(local_labels))
        ])
        Z = self.linkage(mean_embeddings)
        global_labels = self.fcluster(Z, self.threshold)
        labels = self.merge_labels(local_labels, global_labels)

        elapsed = time.time() - s
        print('HAC @ {:.3f}: Clusters {} | Time {:.3f}'.format(
            self.threshold, len(set(labels)), elapsed))

        return labels


    def hac(self, embeddings):

        max_prev_label = 0
        local_labels = []
        for i in range(0, len(embeddings), self.batchsize):

            batch_embeddings = embeddings[ i : i+self.batchsize ]

            Z = self.linkage(batch_embeddings)
            batch_labels = self.fcluster(Z, self.threshold)

            batch_labels += max_prev_label
            max_prev_label += len(set(batch_labels))

            local_labels.extend(batch_labels)

        local_labels = np.array(local_labels)

        mean_embeddings = np.array([
            embeddings[local_labels == y].mean(axis=0)
            for y in sorted(set(local_labels))
        ])

        Z = self.linkage(mean_embeddings)
        global_labels = self.fcluster(Z, self.threshold)

        labels = self.merge_labels(local_labels, global_labels)

        return labels


    def linkage(self, embeddings):

        if self.linkage_method == 'singlefast':
            # Memory-saving algorithm for single linkage
            Z = fastcluster.linkage_vector(
                embeddings, method='single', metric=self.metric)
        else:       
            Z = fastcluster.linkage(
                embeddings, method=self.linkage_method, metric=self.metric)

        return Z


    def fcluster(self, Z, t):
        
        labels = hac.fcluster(Z, t=t, criterion='distance')
        labels -= 1

        return labels
    

    def merge_labels(self, local_labels, global_labels):

        labels = []
        for local in local_labels:
            labels.append(global_labels[local])
        
        return np.array(labels)
