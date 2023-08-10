import numpy as np
import pandas as pd

from typing import Tuple


def pose_angle_distribution(
        angles: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:

    angles = np.array(angles)
    bins = np.arange(-90, 91)

    hist, bins = np.histogram(angles, bins)
    hist = np.append(hist, 0)

    return dict(zip(bins, hist))
