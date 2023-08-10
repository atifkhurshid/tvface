from cnn.arcface import ArcFace


dir = ''
dataset = ''

cfg = {
    'input_shape' : (256, 256, 3),
    'n_classes' : 2000,
    'faces_path' : f'{dir}/{dataset}/faces',
    'data_path' : f'{dir}/{dataset}/metadata/data.csv',
    'prediction_path' : f'{dir}/{dataset}/model/arcface_classifier_weights.h5',
    'batch_size' : 20,
    'threshold' : 0.6,
    'annotations_path' : f'{dir}/{dataset}/metadata/annotations_predicted.json',
}

print('Loading model ...')
af = ArcFace(
    shape=cfg['input_shape'],
    n_classes=cfg['n_classes']
)
af.load(prediction_path=cfg['prediction_path'])

print('Annotating data ...')
af.annotate(
    df_path=cfg['data_path'],
    faces_path=cfg['faces_path'],
    annotations_path=cfg['annotations_path'],
    batch_size=cfg['batch_size'],
    threshold=cfg['threshold'],
)