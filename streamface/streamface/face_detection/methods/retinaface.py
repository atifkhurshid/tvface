import torch
import numpy as np
from face_detection import build_detector


class RetinaFaceDetector(object):

    def __init__(self):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = build_detector(
            'RetinaNetMobileNetV1', device=self.device, clip_boxes=True)
    

    def detect(self, image):
        image = np.expand_dims(image, 0)
        detections = self.detector.batched_detect_with_landmarks(image)
        detections = self.format_detections(detections)

        return detections


    def format_detections(self, detections):
        # Process detections into the format expected by FaceDetection
        formatted_detections = []
        boxlist, landmarklist = detections
        for box, landmarks in zip(boxlist[0], landmarklist[0]):
            confidence = box[4]
            
            X, Y, X2, Y2 = box[:4].astype(int)
            W, H = (X2 - X, Y2 - Y)
            
            names = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
            landmarks = {name : tuple(coords) for name, coords in zip(names, landmarks.astype(int))}

            formatted_detections.append({
                'confidence' : confidence,
                'box' : [X, Y, W, H],
                'landmarks' : landmarks,
            })

        return formatted_detections
