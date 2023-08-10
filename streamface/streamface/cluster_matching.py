import numpy as np

from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import pairwise_distances

from .utils import pkread, jread, jwrite


class ClusterMatching(object):
    """Find similar clusters"""

    def __init__(
            self, features_path, annotations_path, evals_path,
            metric, topk, min_threshold, max_threshold, mode='average'):

        self.features_path = features_path
        self.annotations_path = annotations_path
        self.evals_path = evals_path
        self.metric = metric
        self.topk = topk
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.mode = mode


    def match(self):

        print('Reading data ...')
        embeddings, labels, ignore = self.readdata()

        print('Analyzing clusters ...')
        matches = self.match_clusters(
            embeddings, labels)

        print('Removing ignored entries ...')
        for k in ignore:
            matches.pop(k, None)

        print(f'Writing evaluation to {self.evals_path} ...')
        eval = {
            'matches' : matches,
        }
        jwrite(self.evals_path, eval)


    def readdata(self):

        features = pkread(self.features_path)

        annotations = jread(self.annotations_path)

        embeddings, labels = self._aligndata(features, annotations)

        assert len(embeddings) == len(labels)

        try:
            ignore = annotations['ignore']
        except:
            ignore = []
        
        return embeddings, labels, ignore


    def _aligndata(self, features, annotations):

        annotations = annotations['labels']

        embeddings, labels = [], []
        for name in annotations.keys():
            try:
                emb = features[name + '.png'][0]
                label = int(annotations[name]['label'])

                embeddings.append(emb)
                labels.append(label)
            except:
                continue
        
        return np.array(embeddings), np.array(labels)


    def match_clusters(self, embeddings, labels):

        clean_labels = labels[labels >= 0]
        labels_set = set(clean_labels)
        labels_set = np.array(sorted(labels_set))
        labels_set = labels_set[labels_set <= self.topk]

        matches = self.get_cluster_matches(embeddings, labels, labels_set)

        return matches


    def get_cluster_matches(self, embeddings, labels, labels_set):

        if self.mode == 'average':

            means = np.array([
                embeddings[labels == y].mean(axis=0)
                for y in labels_set
            ])
            dst = pairwise_distances(means, metric=self.metric)
            dst = np.triu(dst)

        elif self.mode == 'single':

            dst = np.full((len(labels_set), len(labels_set)), np.inf)
            for i in tqdm(range(len(labels_set))):
                embeddings_i = embeddings[labels == labels_set[i]]
                for j in range(i + 1, len(labels_set)):
                    embeddings_j = embeddings[labels == labels_set[j]]
                    dst[i][j] = pairwise_distances(
                        embeddings_i,
                        embeddings_j,
                        metric=self.metric,
                    ).min()

        elif self.mode == 'triplet':

            centers = np.array([
                embeddings[labels == y].mean(axis=0)
                for y in labels_set
            ])

            edges1 = []
            edges2 = []
            for y in labels_set:
                embeddings_y = embeddings[labels == y]
                internal_dst = pairwise_distances(
                    embeddings_y,
                    metric=self.metric
                )
                edges = np.unravel_index(
                    np.argmax(
                        internal_dst, axis=None
                    ),
                    internal_dst.shape
                )
                edges1.append(embeddings_y[edges[0]])
                edges2.append(embeddings_y[edges[1]])
            edges1 = np.array(edges1)
            edges2 = np.array(edges2)

            embedding_sets = [centers, edges1, edges2]

            dsts = []
            for embs1 in embedding_sets:
                for embs2 in embedding_sets:
                    dsts.append(
                        pairwise_distances(
                            embs1,
                            embs2,
                            metric=self.metric,
                        )
                    )
            dsts = np.stack(dsts, axis=-1)
            dst = np.min(dsts, axis=-1)

        else:
            print(f"Error: analyze_mode must be 'average', 'single' or 'triplet'")

        dst[dst > self.max_threshold ] = np.inf
        dst[dst == 0.] = np.inf

        indices = np.unravel_index(np.argsort(dst, axis=None), dst.shape)
        pairs = np.column_stack(indices)

        matches_dict = defaultdict(dict)
        for x, y in pairs:
            if dst[x, y] <= self.max_threshold:
                val = round(float(dst[x, y]), 4)
                matches_dict[labels_set[x]][labels_set[y]] = val
            else:
                break

        # For each cluster, keep matches only if the closest match is within min_threshold
        matches_dict = {
            k: v for k, v in matches_dict.items()
            if v[list(v.keys())[0]] < self.min_threshold    # if value of first key
        }

        # Sort by cluster id
        matches_dict = dict(sorted(
            matches_dict.items(),
            key=lambda item: item[0]
        ))

        # Convert dict of dicts to single dict
        matches_dict = {
            f'{k1},{k2}': v for k1, d in matches_dict.items() for k2, v in d.items()
        }

        return matches_dict
