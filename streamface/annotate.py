from streamface.fiftyone_annotation import FiftyOneAnnotation


model = FiftyOneAnnotation(
    name='skynews',
    faces_path='./data/skynews/faces',
    evals_path='./data/skynews/metadata/cluster_evals.json',
    annotations_path='./data/skynews/metadata/annotations.json',
    new_annotations_path='./data/skynews/metadata/annotations_manual.json',
)

model.annotate()
