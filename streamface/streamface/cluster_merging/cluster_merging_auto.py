import numpy as np

from sklearn.metrics import pairwise_distances

from ..utils import Graph


class ClusterMergingAuto(object):

    def __init__(self, metric, threshold):
        self.metric = metric
        self.threshold = threshold


    def mergelabels(self, filenames, embeddings, labels, matches):

        embeddings = np.array(embeddings)

        labels_graph = self.create_matches_graph(
            filenames, embeddings, labels, matches)

        merged_labels = self.merge_matched_labels(labels_graph, labels)

        labels = self.remove_label_gaps(merged_labels)

        return labels


    def create_matches_graph(self, filenames, embeddings, labels, matches):

        annotations = {k:v for k, v in zip(filenames, labels)}

        graph = Graph(len(set(labels)))

        for name1, name2 in matches.items():
            try:
                label1 = annotations[name1]
                label2 = annotations[name2]
                if label1 != label2:
                    emb1 = embeddings[labels == label1]
                    emb2 = embeddings[labels == label2]
                    dst_mat = pairwise_distances(emb1, emb2, metric=self.metric)
                    mean_dst = dst_mat.mean(axis=0).mean()
                    if mean_dst < self.threshold:
                        graph.addEdge(label1, label2)
            except:
                continue

        return graph


    def merge_matched_labels(self, graph, labels):

        groups = graph.connectedComponents()

        merged_labels = labels.copy()
        for group in groups:
            if len(group) > 1:
                for y in group[1:]:
                    merged_labels[labels == y] = group[0]
        
        return merged_labels


    def remove_label_gaps(self, gap_labels):

        filled_labels = gap_labels.copy()
        sorted_labels = sorted(set(gap_labels))
        prev = sorted_labels[0] - 1
        for label in sorted_labels:
            gap = label - prev - 1
            if gap:
                filled_labels[gap_labels == label] -= gap
            prev += 1
        
        return filled_labels
