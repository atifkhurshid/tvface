import numpy as np

from pathlib import Path

from streamface.utils import jread, jwrite


def remove_infrequent_clusters(annotations, min_frequency=10):

    labels = np.array([
        int(annotation['label']) for name, annotation in annotations['labels'].items()
    ])

    clean_labels = labels[labels >= 0]

    unique_labels, frequency = np.unique(clean_labels, return_counts=True)
    infrequent_labels = unique_labels[frequency < min_frequency]

    updated_labels = labels.copy()
    for y in infrequent_labels:
        updated_labels[labels == y] = -y

    for name, label in zip(annotations['labels'], updated_labels):
        annotations['labels'][name]['label'] = int(label)
    
    return annotations


if __name__ == "__main__":

    dir = './data'
    names = [
        'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
        'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
        'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
        'rtnews', 'skynews', 'trtworld', 'wion'
    ]

    for i, name in enumerate(names):
        print('Processing {}: '.format(name), end='', flush=True)
        try:
            annotations_path = Path(dir) / name / 'metadata' / 'annotations_manual_new.json'
            annotations = jread(annotations_path)
            annotations = remove_infrequent_clusters(annotations, min_frequency=10)
            jwrite(annotations_path, annotations)
            print('SUCCESS')
        except:
            print('FAILED')
    