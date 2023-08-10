import json
import pickle
import numpy as np

from pathlib import Path


class GCNDatasetPreparation:
        
    def __init__(self, features_path, annotations_path, output_dir, n_classes, train_size=0.5):

        filenames, embeddings, labels = self._readdata(features_path, annotations_path)

        # Exclude unselected classes
        filenames = filenames[(labels >= 0) & (labels < n_classes)]
        embeddings = embeddings[(labels >= 0) & (labels < n_classes)]
        labels = labels[(labels >= 0) & (labels < n_classes)]

        filenames_train, filenames_test, embeddings_train, embeddings_test, labels_train, labels_test = self._train_test_split(
            filenames, embeddings, labels=labels, train_size=train_size)

        cluster_labels_train = self._standardize_labels(labels_train)
        cluster_labels_test = self._standardize_labels(labels_test)

        self._save_dataset(
                output_dir, filenames_train, embeddings_train, cluster_labels_train,
                filenames_test, embeddings_test, cluster_labels_test)


    def _readdata(self, features_path, annotations_path):

        features = self._pkread(features_path)

        annotations = self._jread(annotations_path)

        filenames, embeddings, labels = self._aligndata(features, annotations)

        assert len(filenames) == len(embeddings) == len(labels)

        return filenames, embeddings, labels


    def _pkread(self, filepath):
        with open(filepath, 'rb') as fp:
            obj = pickle.load(fp)
        return obj


    def _jread(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as fp:
            obj = json.load(fp)
        return obj


    def _aligndata(self, features, annotations):

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


    def _train_test_split(self, *arrays, labels=None, train_size=0.5, random_state=None):

        labels_set = np.array((sorted(set(labels))))
        np.random.default_rng(random_state).shuffle(labels_set)

        split_index = int(np.ceil(train_size * len(labels_set)))
        train_labels = labels_set[:split_index]
        test_labels = labels_set[split_index:]

        train_indices = np.isin(labels, train_labels)
        test_indices = np.isin(labels, test_labels)

        res = []
        for i in range(len(arrays)):
            res.append(arrays[i][train_indices])
            res.append(arrays[i][test_indices])
        res.append(labels[train_indices])
        res.append(labels[test_indices])

        return res


    def _standardize_labels(self, raw_labels):

        std_labels = raw_labels.copy()
        sorted_labels = sorted(set(raw_labels))
        for i, y in enumerate(sorted_labels):
            std_labels[raw_labels == y] = i

        return std_labels


    def _save_dataset(
            self, output_dir, filenames_train, embeddings_train, labels_train,
            filenames_test, embeddings_test, labels_test):

        root_path = Path(output_dir)

        filenames_path = root_path / 'filenames'
        filenames_path.mkdir(parents=True, exist_ok=True)
        self._write_items(filenames_train, filenames_path / 'train.txt')
        self._write_items(filenames_test, filenames_path / 'test.txt')

        embeddings_path = root_path / 'features'
        embeddings_path.mkdir(parents=True, exist_ok=True)
        embeddings_train.tofile(embeddings_path / 'train.bin')
        embeddings_test.tofile(embeddings_path / 'test.bin')

        labels_path = root_path / 'labels'
        labels_path.mkdir(parents=True, exist_ok=True)
        self._write_items(labels_train, labels_path / 'train.meta')
        self._write_items(labels_test, labels_path / 'test.meta')


    def _write_items(self, items, filepath):
        with open(filepath, 'w') as fp:
            for item in items:
                fp.write(f'{item}\n')
