import numpy as np
import pandas as pd

from typing import Tuple
from datetime import datetime


def collection_over_time(
        timestamps: pd.Series,
        unit: str = 'days',
    ) -> Tuple[list, list]:
    
    timestamps_transformed = clip_timestamps(timestamps, unit)
    bins, hist = timestamps_to_histogram(timestamps_transformed)

    return dict(zip(bins, hist))
    

def clip_timestamps(
        timestamps: pd.Series,
        unit: str = 'days',
    ) -> pd.Series:

    if unit == 'seconds':
        timestamps_transformed = timestamps.str[:14].map(lambda x: datetime.strptime(x, '%Y%m%d%H%M%S'))
    elif unit == 'minutes':
        timestamps_transformed = timestamps.str[:12].map(lambda x: datetime.strptime(x, '%Y%m%d%H%M'))
    elif unit == 'hours':
        timestamps_transformed = timestamps.str[:10].map(lambda x: datetime.strptime(x, '%Y%m%d%H'))
    elif unit == 'days':
        timestamps_transformed = timestamps.str[:8].map(lambda x: datetime.strptime(x, '%Y%m%d'))
    elif unit == 'months':
        timestamps_transformed = timestamps.str[:6].map(lambda x: datetime.strptime(x, '%Y%m'))
    elif unit == 'years':
        timestamps_transformed = timestamps.str[:4].map(lambda x: datetime.strptime(x, '%Y'))
    
    return timestamps_transformed


def timestamps_to_histogram(
        timestamps: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:

    x, y = zip(*timestamps.value_counts().items())
    x = np.array(x)
    y = np.array(y)
    y = y[x.argsort()]
    x = x[x.argsort()]

    return (x, y)
    