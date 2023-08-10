import numpy as np
from streamface.feature_evaluation import FeatureEvaluation


evaluator = FeatureEvaluation(
    name='skynews',
    features_path='./data/skynews/features/skynews_frame_20230525100138412614_face_0.pkl',
    metric='cosine',
    max_samples=30000,
    k=5000,
    thresholds=list(np.arange(0.1, 0.4, 0.025))
)

evaluator.evaluate()
