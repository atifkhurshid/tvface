import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


def prepare_training_dataset(annotations_path, output_dir, n_classes, test_split=0.2):

    filenames, labels = _read_annotations(annotations_path)

    # Exclude unselected classes (-1 to leave room for noisy class) 
    filenames = filenames[labels < n_classes - 1]
    labels = labels[labels < n_classes - 1]

    # Seperate unlabeled samples
    noisy_filenames = filenames[labels < 0]
    noisy_labels = labels[labels < 0]

    # Give last label to noisy class
    labels[labels < 0] = n_classes - 1

    filenames_train, filenames_test, labels_train, labels_test = train_test_split(
        filenames, labels, test_size=test_split, stratify=labels,
    )

    df_train = pd.DataFrame({
        'filename' : filenames_train,
        'class' : labels_train,
    })

    df_test = pd.DataFrame({
        'filename' : filenames_test,
        'class' : labels_test,
    })

    df_noisy = pd.DataFrame({
        'filename' : noisy_filenames,
        'class' : noisy_labels,
    })

    train_path = Path(output_dir) / 'train.csv'
    test_path = Path(output_dir) / 'test.csv'
    noisy_path = Path(output_dir) / 'noisy.csv'

    df_train.to_csv(train_path, index=None)
    df_test.to_csv(test_path, index=None)
    df_noisy.to_csv(noisy_path, index=None)


def _read_annotations(annotations_path):

    annotations = _jread(annotations_path)['labels']
    filenames, labels = [], []
    for name, annotation in annotations.items():
        filenames.append(name + '.png')
        labels.append(int(annotation['label']))
    filenames = np.array(filenames)
    labels = np.array(labels)

    return filenames, labels


def _jread(filepath):
    with open(filepath, 'r', encoding='utf-8') as fp:
        obj = json.load(fp)
    return obj


###########################################################################


def prepare_new_dataset(faces_path, output_path):

    filenames = _getfilenames(faces_path, ['.png'])

    df = pd.DataFrame({
        'filename' : filenames,
        'class' : np.full((len(filenames),), fill_value=-1),
    })

    df.to_csv(output_path, index=None)


def _getfilenames(path, exts):

    def is_extfile(fp):
        ret = True if fp.is_file() and fp.suffix in exts else False
        return ret

    filenames = sorted(
        [fp.name for fp in Path(path).iterdir() if is_extfile(fp)],
        key=lambda x: Path(x).stem)

    return filenames
