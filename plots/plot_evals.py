import numpy as np

import matplotlib.pyplot as plt

from collections import defaultdict
from matplotlib import gridspec

from streamface.utils import pkread


plt.style.use('seaborn-paper')

def grid_dimensions(N):
    cols = np.ceil(np.sqrt(N))
    rows = np.ceil(N / cols)
    return int(rows), int(cols)


def score_vs_threshold(scores_dict, thresholds, title='Title'):

    inverted_scores_dict = defaultdict(list)
    for name, scores in scores_dict.items():
        for metric, value in scores.items():
            if not isinstance(value, np.ndarray):
                inverted_scores_dict[metric].append(value)

    rows, cols = grid_dimensions(len(inverted_scores_dict))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()

    for i, (metric, values) in enumerate(inverted_scores_dict.items()):
        ax = fig.add_subplot(gs[i])
        ax.plot(thresholds, values, marker='o')
        ax.set_xticks(thresholds)
        ax.set(ylabel=metric)

    fig.suptitle(title)
    fig.tight_layout()
    # plt.savefig(title, dpi=500)
    plt.show()


def silhouette_distribution(scores_dict, title='Title', xlabel='x', ylabel='y', loc='upper left'):
    rows, cols = grid_dimensions(len(scores_dict))
    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    ax = None
    for i, (name, scores) in enumerate(scores_dict.items()):
        if ax:
            ax = fig.add_subplot(gs[i], sharex=ax, sharey=ax)
        else:
            ax = fig.add_subplot(gs[i])
        
        y = scores['Silhouette PS']
        bins = [-1., -.9, -.8, -.7, -.6, -.5, -.4, -.3, -.2, -.1,
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        y, _ = np.histogram(y, bins)
        plt.bar(bins[:-1], y, width=0.07, align='edge', label=name)

        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(loc=loc)

    fig.suptitle(title)
    fig.tight_layout()
    # plt.savefig(title, dpi=500)
    plt.show()


def read_scores(filenames):
    scores = {}
    for name, filepath in filenames.items():
        scores[name] = pkread(filepath)
    
    return scores


if __name__ == '__main__':
    directory = '../data/TEST/testcluster/metadata/'
    filenames = {
        'OHC 0.5' : directory + 'sky_ohc_5.pkl',
        'OHC 0.55' : directory + 'sky_ohc_45.pkl',
        'OHC 0.6' : directory + 'sky_ohc_4.pkl',
        'OHC 0.65' : directory + 'sky_ohc_35.pkl',
        'HAC2 0.85-0.65' : directory + 'sky_hc_15_35.pkl',
    }

    scores_dict = read_scores(filenames)

    silhouette_distribution(
        scores_dict,
        title='Distribution of per sample Silhouette scores',
        xlabel='Silhouette score',
        ylabel='Number of datapoints'
    )

    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.66]
    score_vs_threshold(
        scores_dict,
        thresholds,
        title='HAC Cosine - Unsupervised metrics at different thresholds'
    )
    