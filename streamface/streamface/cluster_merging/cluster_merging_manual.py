import numpy as np

from ..utils import Graph


class ClusterMergingManual(object):

    def __init__(self):
        pass


    def update_annotations(self, annotations, labels_map, ignore, final=False):

        if len(labels_map):

            labels = self.extract_labels(annotations['labels'])

            clean_labels = labels[labels >= 0]

            groups, noisy = self.get_connected_label_groups(clean_labels, labels_map)

            clean_labels = self.merge_labels(clean_labels, groups, noisy)

            if final:

                clean_labels = self.remove_label_gaps(clean_labels)

                clean_labels = self.order_labels_by_frequency(clean_labels)

            labels[labels >= 0] = clean_labels

            self.overwrite_labels(annotations['labels'], labels)

            msg = 'Report\n'
            msg += f'\tClusters: {len(set(labels))}\n'
            msg += f'\tSamples: {len(labels)}\n'
            msg += f'\tClean: {len(labels[labels != -1])}\n'
            msg += f'\tNoisy: {len(labels[labels == -1])}'
            print(msg)

        if final:
            annotations.pop('ignore', None)
        else:
            try:
                annotations['ignore'].extend(ignore)
            except:
                annotations['ignore'] = ignore

        return annotations


    def extract_labels(self, annotations):

        labels = np.array([
            int(annotation['label']) for name, annotation in annotations.items()
        ])

        return np.array(labels)


    def get_connected_label_groups(self, labels, labels_map):

        labels_set = set(labels)

        graph = Graph(max(labels_set) + 1)
        noisy = []

        for (label1, label2) in labels_map:
            label1 = int(label1)
            label2 = int(label2)
            if label1 != label2:
                if label2 == -1:
                    noisy.append(label1)
                else:
                    graph.addEdge(label1, label2)

        groups = [x for x in graph.connectedComponents() if len(x) > 1]

        return groups, noisy


    def merge_labels(self, labels, groups, noisy):

        merged_labels = labels.copy()

        for y in noisy:
            merged_labels[labels == y] = -1

        for group in groups:
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


    def order_labels_by_frequency(self, labels):

        clean_labels = labels[labels >= 0]

        unique_elements, frequency = np.unique(clean_labels, return_counts=True)
        sorted_indexes = np.argsort(frequency)[::-1]
        sorted_by_freq = unique_elements[sorted_indexes]

        sorted_labels = labels.copy()
        for i, y in enumerate(sorted_by_freq):
            sorted_labels[labels == y] = i

        return sorted_labels


    def overwrite_labels(self, annotations, labels):

        for name, label in zip(annotations, labels):
            annotations[name]['label'] = int(label)
