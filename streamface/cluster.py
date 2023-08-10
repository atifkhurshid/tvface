from streamface.face_clustering import FaceClustering


cluster = FaceClustering(
    features_path='./data/skynews/features/skynews_frame_20230525100138412614_face_0.pkl',
    matches_path='./data/skynews/metadata/matches.csv',
    faces_dir='./data/skynews/faces',
    output_dir='./data/skynews',
    metric='cosine',
    linkage='complete',
    matching_threshold=0.3,
    matching_batchsize=20000,
    merging_threshold=0.5,
)

cluster.call()
