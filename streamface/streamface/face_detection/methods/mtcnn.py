from mtcnn import MTCNN


class MTCNNDetector(object):
    def __init__(self):
        self.detector = MTCNN()
    
    def detect(self, image):
        detections = self.detector.detect_faces(image)
        detections = self.format_detections(detections)
        return detections
    
    def format_detections(self, detections):
        # Process detections into the format expected by FaceDetection
        formated_detections = []
        for detection in detections:
            formated_detections.append({
                'confidence' : detection['confidence'],
                'box' : detection['box'],
                'landmarks' : detection['keypoints'],
            })

        return formated_detections
