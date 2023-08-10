from tqdm import tqdm
from pathlib import Path

from dataset import GCNDatasetPreparation
from dataset import prepare_cnn_dataset


dir = Path('D:/StreamFace/data')
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]

for name in tqdm(names):

    prepare_cnn_dataset(
        annotationsdf_path = dir / name / 'dataset.csv',
        output_dir = dir / name / 'dataset' / 'cnn',
        n_classes = 10000,
        test_size = 0.2,
    )

    features_dir = dir / name / 'features'
    features_path = list(features_dir.iterdir())[0]
    GCNDatasetPreparation(
        features_path=features_path,
        annotations_path = dir / name / 'metadata' / 'annotations_manual_new.json',
        output_dir = dir / name / 'dataset' / 'gcn',
        n_classes = 10000,
        train_size = 0.5,
    )
