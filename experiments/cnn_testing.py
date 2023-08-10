import pandas as pd

from cnn.arcface import ArcFace


dir = 'D:/StreamFace/data'

names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]
names = ['abcnews', 'abcnewsaus', 'africanews', 'arirang', 'wion']

for dataset in names:

    print(dataset)

    cfg = {
        'input_shape' : (256, 256, 3),
        'faces_path' : f'{dir}/{dataset}/faces',
        'test_path' : f'{dir}/{dataset}/dataset/cnn/test.csv',
        'prediction_path' : f'{dir}/{dataset}/model/arcface_classifier_weights.h5',
        'batch_size' : 64,
    }

    test_df = pd.read_csv(cfg['test_path'])
    cfg['n_classes'] = test_df['class'].max() + 1


    print('Loading model ...')
    af = ArcFace(
        shape=cfg['input_shape'],
        n_classes=cfg['n_classes']
    )
    af.load(prediction_path=cfg['prediction_path'])

    print('Testing model ...')
    af.test(
        df=test_df,
        faces_path=cfg['faces_path'],
        batch_size=cfg['batch_size'],
    )
