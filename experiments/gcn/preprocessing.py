import json
import pickle
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare_clustering_dataset(features_path, annotations_path, output_dir, test_split=0.5):

    filenames, embeddings, labels = _readdata(features_path, annotations_path)

    # Give last label to noisy class
    labels[labels < 0] = len(labels)

    filenames_train, filenames_test, embeddings_train, embeddings_test, labels_train, labels_test = train_test_split(
        filenames, embeddings, labels, test_size=test_split)

    labels_train = _standardize_labels(labels_train)
    labels_test = _standardize_labels(labels_test)

    filenames_path = Path(output_dir) / 'filenames'
    filenames_path.mkdir(parents=True, exist_ok=True)
    _write_items(filenames_train, filenames_path / 'train.txt')
    _write_items(filenames_test, filenames_path / 'test.txt')

    embeddings_path = Path(output_dir) / 'features'
    embeddings_path.mkdir(parents=True, exist_ok=True)
    embeddings_train.tofile(embeddings_path / 'train.bin')
    embeddings_test.tofile(embeddings_path / 'test.bin')

    labels_path = Path(output_dir) / 'labels'
    labels_path.mkdir(parents=True, exist_ok=True)
    _write_items(labels_train, labels_path / 'train.meta')
    _write_items(labels_test, labels_path / 'test.meta')


def _readdata(features_path, annotations_path):

    features = _pkread(features_path)

    annotations = _jread(annotations_path)

    filenames, embeddings, labels = _aligndata(features, annotations)

    assert len(filenames) == len(embeddings) == len(labels)

    return filenames, embeddings, labels


def _pkread(filepath):
    with open(filepath, 'rb') as fp:
        obj = pickle.load(fp)
    return obj


def _jread(filepath):
    with open(filepath, 'r', encoding='utf-8') as fp:
        obj = json.load(fp)
    return obj


def _aligndata(features, annotations):

    annotations = annotations['labels']

    filenames, embeddings, labels = [], [], []
    for name in annotations.keys():
        try:
            emb = features[name + '.png'][0]
            label = int(annotations[name]['label'])
        except:
            continue
        filenames.append(name)
        embeddings.append(emb)
        labels.append(label)
    
    return np.array(filenames), np.array(embeddings), np.array(labels)


def _standardize_labels(raw_labels):

    std_labels = raw_labels.copy()
    sorted_labels = sorted(set(raw_labels))
    for i, y in enumerate(sorted_labels):
        std_labels[raw_labels == y] = i

    return std_labels


def _write_items(items, filepath):
    with open(filepath, 'w') as fp:
        for item in items:
            fp.write(f'{item}\n')


if __name__ == '__main__':

    dir = '../../data/sky'
    config = [
        f'{dir}/features/skynews_frame_20211212205424059632_face_0.pkl',
        f'{dir}/metadata/annotations_manual.json',
        f'{dir}/gcn/',
        0.8,
    ]

    prepare_clustering_dataset(
        features_path=config[0],
        annotations_path=config[1],
        output_dir=config[2],
        test_split=config[3],
    )
