from streamface.annotation_evaluation import AnnotationEvaluation


evaluator = AnnotationEvaluation(
    true_annotations_path='./data/skynews/metadata/annotations_manual.json',
    pred_annotations_path='./data/skynews/metadata/annotations.json',
)

score = evaluator.evaluate(verbose=False)

print(score)
