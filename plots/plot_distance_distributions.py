import functools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import pairwise_distances_chunked
from streamface.streamface.utils import pkread, jread


def readdata(features_path, annotations_path):

    def aligndata(features, annotations):

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


    features = pkread(features_path)

    annotations = jread(annotations_path)

    embeddings, labels = aligndata(features, annotations)

    assert len(embeddings) == len(labels)

    return embeddings, labels


def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    # accumulate distances from each sample to each cluster
    clust_dists = np.zeros((len(D_chunk), len(label_freqs)), dtype=D_chunk.dtype)
    for i in range(len(D_chunk)):
        clust_dists[i] += np.bincount(
            labels, weights=D_chunk[i], minlength=len(label_freqs)
        )

    # intra_index selects intra-cluster distances within clust_dists
    intra_index = (np.arange(len(D_chunk)), labels[start : start + len(D_chunk)])
    # intra_clust_dists are averaged over cluster size outside this function
    intra_clust_dists = clust_dists[intra_index]
    # of the remaining distances we normalise and extract the minimum
    clust_dists[intra_index] = np.inf
    clust_dists /= label_freqs
    inter_clust_dists = clust_dists.min(axis=1)
    return intra_clust_dists, inter_clust_dists


def silhouette_distances(X, labels, metric='cosine', **kwds):
    label_freqs = np.bincount(labels)

    kwds["metric"] = metric
    reduce_func = functools.partial(
        _silhouette_reduce, labels=labels, label_freqs=label_freqs
    )
    results = zip(*pairwise_distances_chunked(X, reduce_func=reduce_func, **kwds))
    intra_clust_dists, inter_clust_dists = results
    intra_clust_dists = np.concatenate(intra_clust_dists)
    inter_clust_dists = np.concatenate(inter_clust_dists)

    denom = (label_freqs - 1).take(labels, mode="clip")
    with np.errstate(divide="ignore", invalid="ignore"):
        intra_clust_dists /= denom

    return intra_clust_dists, inter_clust_dists


def histogram_of_distances(dsts):
    hist, bins = np.histogram(dsts, bins=np.linspace(-.01, 2.01, 102), density=True)
    bins = np.around((bins[:-1] + bins[1:]) / 2, 2)

    return hist, bins



root_dir = Path("D:/StreamFace/data")
name = ''
features_path = ""
annotations_path = ""
df_path = root_dir / name / 'dataset50_processed.csv'
output_path = root_dir / name / 'dataset50_distance_distribution.csv'

"""
embeddings, labels = readdata(features_path, annotations_path)
intra, inter = silhouette_distances(embeddings, labels)
"""

df = pd.read_csv(df_path)

mean_dsts = df['distance_from_mean']
intra_dsts = df['intra_class_distance']
inter_dsts = df['nearest_class_distance']

mean_hist, bins = histogram_of_distances(mean_dsts)
intra_hist, bins = histogram_of_distances(intra_dsts)
inter_hist, bins = histogram_of_distances(inter_dsts)

hist_df = pd.DataFrame({
    'cosine_distance': bins,
    'distance_from_mean' : mean_hist,
    'intra_class_distance': intra_hist,
    'nearest_class_distance': inter_hist,
})

hist_df.to_csv(output_path, index=False)

print()