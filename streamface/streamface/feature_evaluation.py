import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from scipy.interpolate import interp1d
from matplotlib import gridspec

from .utils import pkread

plt.style.use('seaborn-paper')


class FeatureEvaluation(object):
    """Evaluate face features"""

    def __init__(self, name, features_path, metric, max_samples, k, thresholds):

        self.name = name
        self.features_path = features_path
        self.metric = metric
        self.max_samples = max_samples
        self.k = k
        self.thresholds = thresholds


    def evaluate(self):

        print('Reading data ...')
        embeddings = self.readdata()

        print('Evaluating ...')

        self.plot_pairwise_distance_distribution(embeddings, self.metric)

        self.plot_thresholded_distance_distribution(embeddings, self.metric)


    def readdata(self):

        features = pkread(self.features_path)

        embeddings, _, _, _, _= zip(*list(features.values()))

        embeddings = np.array(embeddings)

        if len(embeddings) > self.max_samples:
            rng = np.random.default_rng()
            indices = rng.choice(len(embeddings), size=self.max_samples, replace=False)
            embeddings = embeddings[indices]

        return embeddings


    def plot_pairwise_distance_distribution(self,
            features, metric='cosine', num_bins=50, range=(0, 2), alpha=10,
            title='Title', xlabel='x', ylabel='y', loc='best'):

        title = 'Pairwise distance distribution in embedding space'
        xlabel = self.metric + ' distance'
        ylabel = 'Probability'

        fig, ax = plt.subplots()

        bins = np.linspace(range[0], range[1], num_bins, endpoint=True)
        distances_dict = self._get_knn_distances(features, metric, self.k)

        for name, distances in distances_dict.items():

            hist, _ = np.histogram(distances, bins, density=True)
            hist = hist / hist.sum()
            midpoints = (bins[1:] + bins[:-1]) / 2

            ax.bar(midpoints, hist, width=0.9*(midpoints[2] - midpoints[1]),
                    align='center', alpha=0.4)

            f = interp1d(midpoints, hist, kind='quadratic')
            x = np.linspace(midpoints.min(), midpoints.max(), int(alpha*num_bins), endpoint=True)
            
            ax.plot(x, f(x), lw=2, label=name)

        ax.set_xlim(range)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(loc=loc)
        
        plt.tight_layout()
        # plt.savefig(f'Pairwise - {self.name}', dpi=500)
        plt.show()


    def _get_knn_distances(self, embeddings, metric, k):

        distance_matrix = pairwise_distances(embeddings, metric=metric)
        sorted_distance_matrix = np.sort(distance_matrix, axis=1)[:, 1:k]

        distances = {
            'Minimum' : sorted_distance_matrix.min(axis=1),
            'Mean' : sorted_distance_matrix.mean(axis=1),
            'Maximum' : sorted_distance_matrix.max(axis=1),
        }

        return distances


    def plot_thresholded_distance_distribution(self,
            features, metric='cosine', num_bins=50, xrange=(0, 1.5), alpha=10,
            title='Title', xlabel='x', ylabel='y', loc='best'):

        title = 'Thresholded distance distribution in embedding space'
        xlabel = self.metric + ' distance'
        ylabel = 'Probability'

        bins = np.linspace(xrange[0], xrange[1], num_bins, endpoint=True)
        distances_dicts = self._get_threshold_distances(features, metric, self.k, self.thresholds)

        rows, cols = self.grid_dimensions(len(distances_dicts))
        gs = gridspec.GridSpec(rows, cols)
        fig = plt.figure()
        ax = None
        for i, (threshold, distances_dict) in enumerate(distances_dicts.items()):
            if ax:
                ax = fig.add_subplot(gs[i], sharex=ax, sharey=ax)
            else:
                ax = fig.add_subplot(gs[i])

            for name, distances in distances_dict.items():

                hist, _ = np.histogram(distances, bins, density=True)
                hist = hist / hist.sum()
                midpoints = (bins[1:] + bins[:-1]) / 2

                ax.bar(midpoints, hist, width=0.9*(midpoints[2] - midpoints[1]),
                        align='center', alpha=0.4)

                f = interp1d(midpoints, hist, kind='quadratic')
                x = np.linspace(midpoints.min(), midpoints.max(), int(alpha*num_bins), endpoint=True)
                
                ax.plot(x, f(x), lw=2, label=name)

                ax.set(title=threshold, xlim=list(xrange))

            if i == len(distances_dicts) - 1:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower right')

        plt.tight_layout()
        # plt.savefig(f'Threshold - {self.name}', dpi=500)
        plt.show()


    def _get_threshold_distances(self, embeddings, metric, k, thresholds):

        distance_matrix = pairwise_distances(embeddings, metric=metric)
        sorted_distance_matrix_org = np.sort(distance_matrix, axis=1)[:, 1:k]

        distances = {}
        for threshold in thresholds:

            mask = sorted_distance_matrix_org > threshold
            sorted_distance_matrix = np.ma.array(
                sorted_distance_matrix_org, mask=mask)
                
            distances_below = {
                'Minimum-B' : sorted_distance_matrix.min(axis=1).filled(threshold),
                'Mean-B' : sorted_distance_matrix.mean(axis=1).filled(threshold),
                # 'Maximum-B' : sorted_distance_matrix.max(axis=1).filled(threshold),
            }

            mask = sorted_distance_matrix_org <= threshold
            sorted_distance_matrix = np.ma.array(
                sorted_distance_matrix_org, mask=mask)

            distances_above = {
                # 'Minimum-A' : sorted_distance_matrix.min(axis=1).filled(threshold),
                'Mean-A' : sorted_distance_matrix.mean(axis=1).filled(threshold),
                'Maximum-A' : sorted_distance_matrix.max(axis=1).filled(threshold),
            }

            distances[str(round(threshold, 3))] = {**distances_below, **distances_above}

        return distances


    def grid_dimensions(self, N):

        cols = np.ceil(np.sqrt(N))
        rows = np.ceil(N / cols)
        return int(rows), int(cols)
