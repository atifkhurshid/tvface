import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np


dir = Path('./data')
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]

dfs = []
max_class = 0

for name in tqdm(names):

    df = pd.read_csv(dir / name / 'dataset.csv')
    df['filename'] = f'D:/StreamFace/data/{name}/faces/' + df['filename']
    df['class'] += max_class
    max_class = df['class'].max() + 1
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)

sorted_classes = df['class'].value_counts().index.values
labels = df['class'].values
updated_labels = labels.copy()

for i, y in tqdm(enumerate(sorted_classes), total=len(sorted_classes)):
    updated_labels[labels == y] = i

df['class'] = updated_labels

df.to_csv(dir / 'dataset.csv', index=False)

for i in [50, 100]:

    last_class = np.where(df['class'].value_counts() >= i)[0].max()

    df2 = df[df['class'] < last_class]

    df2.to_csv(dir / f'dataset{i}.csv', index=False)
