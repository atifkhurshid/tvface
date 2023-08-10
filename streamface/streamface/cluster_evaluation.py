import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

from .utils import pkread, pkwrite, jread


class ClusterEvaluation(object):
    """Evaluate face clustering"""

    def __init__(self, name, features_path, annotations_path, scores_path, plots_path, metric):

        self.name = name
        self.features_path = features_path
        self.annotations_path = annotations_path
        self.scores_path = scores_path
        self.metric = metric

        self.plots_path = Path(plots_path)
        self.plots_path.mkdir(parents=True, exist_ok=True)


    def evaluate(self):

        print('Reading data ...')
        embeddings, labels = self._readdata()

        embeddings = embeddings[labels >= 0]
        labels = labels[labels >= 0]

        print('Evaluating ...')
        scores = {
            'Davies-Bouldin' : davies_bouldin_score(
                embeddings, labels),
            'Calinski-Harabasz' : calinski_harabasz_score(
                embeddings, labels),
            'Silhouette' : silhouette_score(
                embeddings, labels, metric=self.metric),
            'Silhouette PS' : silhouette_samples(
                embeddings, labels, metric=self.metric),
        }
        for k, v in scores.items():
            if not isinstance(v, np.ndarray):
                print('{} : {:.3f}'.format(k, v))

        print(f'Writing results to {self.scores_path}')
        pkwrite(self.scores_path, scores)

        print('Plotting ...')
        self.plot_class_distribution(labels)
        self.plot_membership_distribution(labels)


    def plot_class_distribution(self, labels, title='Title', xlabel='x', ylabel='y', loc='upper right'):

        title = f'Class Distribution - {self.name}'
        xlabel = 'Class Label'
        ylabel = 'Number of Faces'

        fig, ax = plt.subplots()

        bincounts = np.bincount(labels)
        ax.plot(
            list(range(len(bincounts))),
            bincounts,
            label=self.name
        )
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

        fig.tight_layout()
        plt.savefig(self.plots_path / title, dpi=500)
        #plt.show()


    def plot_membership_distribution(self, labels, title='Title', xlabel='x', ylabel='y', loc='upper right'):

        title = f'Membership Distribution - {self.name}'
        xlabel = 'Number of Faces'
        ylabel = 'Number of Classes'

        fig, ax = plt.subplots()

        labels_frequency = np.bincount(labels)
        maxf = np.max(labels_frequency)

        bins = np.array(sorted([
            1, 2, 6,
            *list(range(10, 99, 10)),
            *list(range(100, 999, 100)),
            *list(range(1000, 9999, 1000)),
            *list(range(10000, 99999, 10000)),
            maxf,
        ]))
        bins = bins[bins <= maxf]

        hist, _ = np.histogram(labels_frequency, bins)
        hist = np.append(hist, 0)
        ax.bar(
            list(range(len(hist))), height=hist,
            width=0.95, align='edge', tick_label=bins,
            label=self.name,
        )
        ax.xaxis.set_tick_params(rotation=90)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

        fig.tight_layout()
        plt.savefig(self.plots_path / title, dpi=500)
        #plt.show()


    def _readdata(self):

        features = pkread(self.features_path)

        annotations = jread(self.annotations_path)

        embeddings, labels = self._aligndata(features, annotations)

        assert len(embeddings) == len(labels)

        return embeddings, labels


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
