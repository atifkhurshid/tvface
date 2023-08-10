from streamface.cluster_evaluation import ClusterEvaluation


evaluator = ClusterEvaluation(
    name='skynews',
    features_path='./data/skynews/features/skynews_frame_20230525100138412614_face_0.pkl',
    annotations_path='./data/skynews/metadata/annotations.json',
    scores_path='./data/skynews/metadata/cluster_scores.pkl',
    plots_path='./data/skynews/metadata',
    metric='cosine',
)

evaluator.evaluate()