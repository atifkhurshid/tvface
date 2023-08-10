import numpy as np
import pandas as pd

from typing import Tuple


def class_membership_distribution(
        labels: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:

    labels = np.array(labels)
    bins = generate_bins()

    hist, bins = np.histogram(labels, bins)

    bins = [f'{bins[i]} - {bins[i + 1] - 1}' for i in range(len(bins) - 1)]

    return dict(zip(bins, hist))


def generate_bins() -> list:

    i = 10
    bins = [0]
    while i <= 1000:
        bins.extend(list(np.arange(i, 10*i, i)))
        i *= 10
    bins.append(10000)

    return bins