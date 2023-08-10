import json
import numpy as np
import pandas as pd


def read_frames(
        filepath: str,
    ) -> pd.DataFrame:

    df = pd.read_csv(filepath)

    return df
    

def read_annotations(
        filepath: str,
    ) -> dict:

    with open(filepath, 'r', encoding='utf-8') as fp:
        obj = json.load(fp)
    
    annotations = obj['labels']

    return annotations


def create_annotations_dataframe(
        annotations: dict,
    ) -> pd.DataFrame:

    names = []
    channels = []
    timestamps = []
    labels = []
    masks = []
    genders = []
    ages = []
    races = []
    expressions = []
    yaws = []
    pitches = []
    rolls = []

    for name, annotation in annotations.items():

        names.append(name)

        channel, _, timestamp, _, _ = name.split('_')
        channels.append(channel)
        timestamps.append(timestamp)

        labels.append(annotation['label'])

        masks.append(_best_class(['Masked', 'Unmasked'], np.array([annotation['mask'], 1-annotation['mask']])))

        genders.append(_best_class(*zip(*annotation['attributes']['gender'].items())))

        ages.append(_best_class(*zip(*annotation['attributes']['age'].items())))

        races.append(_best_class(*zip(*annotation['attributes']['race'].items())))

        expressions.append(_best_class(*zip(*annotation['attributes']['expression'].items())))

        yaws.append(annotation['attributes']['pose']['yaw'])
        pitches.append(annotation['attributes']['pose']['pitch'])
        rolls.append(annotation['attributes']['pose']['roll'])

    df = pd.DataFrame(
        list(zip(names, channels, timestamps, labels, masks, genders, ages, races, expressions, yaws, pitches, rolls)),
        columns=['Name', 'Channel', 'Timestamp', 'Label', 'Mask', 'Gender', 'Age', 'Race', 'Expression', 'Yaw', 'Pitch', 'Roll'])
    
    return df


def _best_class(
        categories: list,
        probabilities: np.ndarray,
    ) -> str:

    return categories[np.argmax(probabilities)]
