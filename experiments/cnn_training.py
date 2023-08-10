import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder

from cnn.arcface import ArcFace


dir = 'D:/StreamFace/data'
# dataset = 'abcnews'

names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]
names = ['abcnews', 'abcnewsaus', 'africanews', 'arirang', 'wion']
names = ['']

for dataset in names:

    print('\n' + '='*50, flush=True)
    print('Dataset: ', dataset, flush=True)
    print('='*50 + '\n', flush=True)

    cfg = {
        'input_shape' : (256, 256, 3),
        'n_classes' : None,
        # 'faces_path' : f'{dir}/{dataset}/faces',
        'faces_path': None,
        # 'train_path' : f'{dir}/{dataset}/dataset/cnn/train.csv',
        'train_path' : f'{dir}/{dataset}/dataset_processed.csv',
        'test_path' : f'{dir}/{dataset}/dataset/cnn/test.csv',
        'validation_split' : 0.05,
        'batch_size' : 64,
        'learning_rate' : 1e-3,
        'epochs' : 5,
        'prediction_path' : f'{dir}/{dataset}/rawmodel/arcface_classifier_weights.h5',
        'embedding_path' : f'{dir}/{dataset}/rawmodel/arcface_weights.h5',
        'verbose': 1,
    }

    train_df = pd.read_csv(cfg['train_path'])

    """
    # Intra Class Distance
    mean_intra_class_distances = train_df.groupby('class', group_keys=False)['intra_class_distance'].mean()
    classes = np.where(mean_intra_class_distances > 0.1)[0]
    train_df = train_df[train_df['class'].isin(classes)] 

    # Face Size
    train_df = train_df[train_df['face_size'] >= 0.05]
    classes, counts = np.unique(train_df['class'], return_counts=True)
    train_df = train_df[train_df['class'].isin(classes[counts >= 5])]
    # """
    """
    # Class membership
    classes, counts = np.unique(train_df['class'], return_counts=True)
    train_df = train_df[train_df['class'].isin(classes[counts >= 50])]
    """

    train_df['class'] = LabelEncoder().fit_transform(train_df['class'])

    cfg['n_classes'] = train_df['class'].max() + 1
    print('Num Classes: ', cfg['n_classes'])
    """
    for model in names:

        print('\n' + '-'*25, flush=True)
        print('Model: ', model, flush=True)
        print('-'*25 + '\n', flush=True)
    """

    print('Creating model ...', flush=True)
    af = ArcFace(
        shape=cfg['input_shape'],
        n_classes=cfg['n_classes']
    )

    # af.load(embedding_path=f'{dir}/{model}/model{i}/arcface_weights.h5')

    print('Training model ...', flush=True)
    af.train(
        df=train_df,
        faces_path=cfg['faces_path'],
        validation_split=cfg['validation_split'],
        batch_size=cfg['batch_size'],
        learning_rate=cfg['learning_rate'],
        epochs=cfg['epochs'],
        verbose=cfg['verbose'],
    )
    # """
    print('Saving model ...', flush=True)
    af.save(
        prediction_path=cfg['prediction_path'],
        embedding_path=cfg['embedding_path'],
    )
    """
    print('Testing model ...', flush=True)
    test_df = pd.read_csv(cfg['test_path'])
    af.test(
        df=test_df,
        faces_path=cfg['faces_path'],
        batch_size=cfg['batch_size'],
    )
    # """