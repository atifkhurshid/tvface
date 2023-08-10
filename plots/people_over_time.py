import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from annotations_processing import read_annotations
from annotations_processing import create_annotations_dataframe
from timestamp_processing import collection_over_time


def plot_people_over_time(
        df: pd.DataFrame,
        labels: list,
        unit: str = 'days',
    ) -> None:

    for label in labels:
        x, y = collection_over_time(df['Timestamp'][df['Label'] == label], unit)
        plt.scatter(x, y, s=y)

    plt.show()


if __name__ == "__main__":

    dir = ''
    name = ''
    labels = [0]

    annotations_path = Path(dir) / name / 'metadata' / 'annotations_manual.json'
    annotations = read_annotations(annotations_path)
    annotations_df = create_annotations_dataframe(annotations)

    plot_people_over_time(annotations_df, labels, 'days')

    