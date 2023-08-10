import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn import datasets

from hri import HierarchicalRetrievalIndexMatching


def call(matcher, data, colors):

    labels, _ = matcher.match(data)

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    for i in range(len(data)):
        ax.scatter(data[i, 0], data[i, 1], c=colors[labels[i]])
    plt.show(block=True)

    matcher.reset()


if __name__ == "__main__":

    np.random.seed(42)

    colors = list(mcolors.XKCD_COLORS)
    np.random.shuffle(colors)

    n_samples = 200

    matcher = HierarchicalRetrievalIndexMatching(metric='euclidean', threshold=0.5)

    data = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05)[0]
    call(matcher, data, colors)

    data = datasets.make_moons(n_samples=n_samples, noise=0.05)[0]
    call(matcher, data, colors)

    data = datasets.make_blobs(n_samples=n_samples, random_state=8)[0] / 10
    call(matcher, data, colors)

    data = np.random.rand(n_samples, 2)
    call(matcher, data, colors)
