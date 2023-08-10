from streamface.cluster_matching import ClusterMatching


matcher = ClusterMatching(
    features_path='./data/skynews/features/skynews_frame_20230525100138412614_face_0.pkl',
    annotations_path='./data/skynews/metadata/annotations.json',
    evals_path='./data/skynews/metadata/cluster_evals.json',
    metric='cosine',
    topk=10000,
    min_threshold=0.5,
    max_threshold=1.0,
    mode='average',
)

matcher.match()
