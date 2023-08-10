import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics.pairwise import pairwise_distances

from streamface.streamface.utils import jread, pkread


def extract_detection_info(dets_df):

    df = dets_df[['name']].copy()
    df = df.rename(columns={'name': 'filename'})
    df['filename'] = df['filename'].str.replace('.png', '.jpg')

    df['frame_height'], df['frame_width'], _ = dets_df['image'].str.strip('()').str.split(',').str
    _, _, df['face_width'], df['face_height'] = dets_df['box'].str.strip('[]').str.split(',').str
    for c in ['frame_height', 'frame_width', 'face_height', 'face_width']:
        df[c] = df[c].astype(int)
    
    face_areas = df['face_height'] * df['face_width']
    frame_areas = df['frame_height'] * df['frame_width']
    df['face_size'] = face_areas / frame_areas

    return df

def extract_annotation_info(annots):

    def _best_class(categories, probabilities):
        return categories[np.argmax(probabilities)]

    names = []
    channels = []
    timestamps = []
    labels = []
    masks = []
    genders = []
    ages = []
    races = []
    expressions = []
    yaws = []
    pitches = []
    rolls = []

    for name, annot in annots.items():

        names.append(name + '.jpg')

        channel, _, timestamp, _, _ = name.split('_')
        channels.append(channel)
        timestamps.append(timestamp)

        labels.append(annot['label'])

        masks.append(_best_class(['Masked', 'Unmasked'], np.array([annot['mask'], 1-annot['mask']])))

        genders.append(_best_class(*zip(*annot['attributes']['gender'].items())))

        ages.append(_best_class(*zip(*annot['attributes']['age'].items())))

        races.append(_best_class(*zip(*annot['attributes']['race'].items())))

        expressions.append(_best_class(*zip(*annot['attributes']['expression'].items())))

        yaws.append(annot['attributes']['pose']['yaw'])
        pitches.append(annot['attributes']['pose']['pitch'])
        rolls.append(annot['attributes']['pose']['roll'])

    df = pd.DataFrame(
        list(zip(names, channels, timestamps, labels, masks, genders, ages, races, expressions, yaws, pitches, rolls)),
        columns=['filename', 'channel', 'timestamp', 'class', 'mask', 'gender', 'age', 'race', 'expression', 'yaw', 'pitch', 'roll'])
    
    return df


def add_feeature_info(df, feats, metric='cosine'):

    import functools
    from sklearn.metrics import pairwise_distances_chunked

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

    embeddings = []
    for name in df['filename']:
        key = name.replace('.jpg', '.png')
        emb = feats[key][0]
        embeddings.append(emb)
    embeddings = np.array(embeddings)

    labels = df['class'].values
    labels_set = np.unique(labels)

    mean_embeddings = np.array([
        embeddings[labels == y].mean(axis=0) for y in labels_set])

    dsts = []
    for emb, label in zip(embeddings, labels):
        dst = pairwise_distances(
            [emb], [mean_embeddings[label]], metric=metric)[0, 0]
        dsts.append(dst)

    intra, inter = silhouette_distances(embeddings, labels, metric)

    df['distance_from_mean'] = dsts
    df['intra_class_distance'] = intra
    df['nearest_class_distance'] = inter
    
    return df


dir = Path('./data')
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]


for name in names:
    print(name)

    dets_path = dir / name / 'metadata' / 'detections.csv'
    dets = pd.read_csv(dets_path)
    dets.drop_duplicates('name', inplace=True, ignore_index=True)
    dets_df = extract_detection_info(dets)

    annots_path = dir / name / 'metadata' / 'annotations_manual_new.json'
    annots = jread(annots_path)['labels']
    annots_df = extract_annotation_info(annots)

    df = pd.merge(annots_df, dets_df, on='filename', how='left')
    df[['frame_height', 'frame_width']] = df[['frame_height', 'frame_width']].fillna(1)
    df[['face_width', 'face_height', 'face_size']] = df[['face_width', 'face_height', 'face_size']].fillna(0)

    feats_dir = dir / name / 'features'
    feats_path = list(feats_dir.iterdir())[0]
    feats = pkread(feats_path)
    df = add_feeature_info(df, feats)

    df = df[['filename', 'channel', 'timestamp', 'frame_height', 'frame_width',
             'face_height', 'face_width', 'face_size', 'distance_from_mean',
             'intra_class_distance', 'nearest_class_distance',
             'class', 'mask', 'gender', 'age', 'race', 'expression',
             'yaw', 'pitch', 'roll']]

    df_path = dir / name / 'dataset.csv'
    df.to_csv(df_path, index=False)
    