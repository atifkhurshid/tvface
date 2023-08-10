import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from streamface.streamface.utils import pkread
import numpy as np
from sklearn.cluster import AgglomerativeClustering


dir = Path('./data')
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]

df = pd.read_csv(dir / 'dataset.csv')

dffeatures = np.zeros((df['class'].max() + 1, 512))
dfclassfrequency = np.zeros(df['class'].max() + 1)
done = np.zeros(df['class'].max() + 1)

for name in names:

    print(name)

    df_name = df[df['filename'].str.contains(name + '_')]

    features_path = list((dir / name / 'features').iterdir())[0]
    features = pkread(features_path)

    embeddings = []
    for dfkey in tqdm(df_name['filename'].values):
        fkey = dfkey.split('/')[-1].replace('jpg', 'png')
        embeddings.append(features[fkey][0])
    
    embeddings = np.array(embeddings)
    labels = np.array(df_name['class'].values)
    classdistribution = df_name['class'].value_counts(sort=False).sort_index()
    classes = classdistribution.index
    frequency = classdistribution.values

    for i, y in enumerate(classes):
        dffeatures[y] = embeddings[labels == y].mean(axis=0)
        dfclassfrequency[y] = frequency[i]
        done[y] = 1

clustering = AgglomerativeClustering(
    n_clusters=None,
    affinity='cosine',
    linkage='single',
    distance_threshold=0.2).fit(dffeatures)
clusterlabels = clustering.labels_

selected_classes = []
for i in np.unique(clusterlabels):
    clustermembers = np.where(clusterlabels == i)[0]
    membersfrequency = dfclassfrequency[clustermembers]
    selected_classes.append(clustermembers[np.argmax(membersfrequency)])

selecteddf = df[df['class'].isin(selected_classes)]
selecteddf.reset_index(drop=True, inplace=True)

newdf = selecteddf.copy()
for i, c in enumerate(sorted(selecteddf['class'].unique())):
    newdf.loc[selecteddf['class'] == c, 'class'] = i

newdf.to_csv("./data/dataset_processed.csv", index=False)

print()