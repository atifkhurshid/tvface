from streamface.face_extraction import FaceExtraction

"""
conf_threshold: Probability of face returned by face detector
size_threshold: Min ratio of face area to frame area (fH * fW) / (imH * imW)     
margin: Fraction of height and width. Adds extra size to face image because
        face detectors return a tightly cropped image
image_size: Size of face images for dataset

"""

extract = FaceExtraction(
    input_dir='./data/skynews',
    output_dir='./data/skynews',
    detection='retinaface',
    batch_size=32,
    conf_threshold=0.95,
    size_threshold=0.005,
    blur_threshold=25,
    match_thresholds=(0.75, 0.75),
    face_size=(256, 256),
    aligned_size=(128, 128),
    padding=0.5,
    margin=1.0,
    resume=True,
    log_interval=100
)

extract.call()
