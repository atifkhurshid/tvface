import numpy as np

from deepface.detectors.FaceDetector import alignment_procedure

from ..utils import imresize
from ..utils import detect_blur
from ..utils import template_matching
from ..utils import mean_boxes_iou


class FaceDetection(object):
    "Detect and align faces in images"

    _detectors = ['mtcnn', 'retinaface']
    
    def __init__(
            self,
            method,
            conf_threshold,
            size_threshold,
            blur_threshold,
            match_thresholds,
            face_size,
            aligned_size,
            padding,
            margin
        ):

        self.conf_threshold = conf_threshold
        self.size_threshold = size_threshold
        self.blur_threshold = blur_threshold
        self.similarity_threshold = match_thresholds[0]
        self.iou_threshold = match_thresholds[1]
        self.face_size = face_size
        self.aligned_size = aligned_size
        self.padding = padding
        self.margin = margin

        self.model = self.getmodel(method)

        self.previous_image = None
        self.previous_dets = []
        self.previous_previous_image = None
        self.previous_previous_dets = []


    @classmethod
    def available_detectors(cls):
        return cls._detectors


    def getfaces(self, image):
        """Detect, crop, and align faces in a given image.

        Args:
            image (ndarray): Image as numpy array

        Returns:
            3-tuple : Dictionary of detection coordinates,
                      List of detected faces,
                      List of cropped and aligned faces
        """
        processed_dets, faces, aligned_faces, match = [], [], [], 0

        detections = self.model.detect(image)

        for det in detections:
            
            processed_det, face, aligned_face = self.process_detection(image, det)

            processed_dets.extend(processed_det)
            faces.extend(face)
            aligned_faces.extend(aligned_face)

        match = self.compare_with_previous(image, processed_dets)

        return (processed_dets, faces, aligned_faces, match)


    def process_detection(self, image, detection):
        """Crop and align a detected face in an image

        Ensures that detection confidence, face area, and image clarity
        are above certain thresholds.

        Args:
            image (ndarray): Image as a numpy array
            detection (dict): Dictionary of bounding box and keypoint coordinates

        Returns:
            3-tuple: Dictionary of detection coordinates,
                     Detected face,
                     Cropped and aligned face 
        """
        X, Y, W, H = detection['box']
        landmarks = detection['landmarks']
        conf = detection['confidence']

        image_shape = image.shape
        px = int(self.padding * image_shape[1])
        py = int(self.padding * image_shape[0])
        image = np.pad(image, ((py, py), (px, px), (0, 0)), mode='constant')
        X = max(0, X) + px
        Y = max(0, Y) + py

        size = (W * H) / (image_shape[0] * image_shape[1])

        if conf > self.conf_threshold and size > self.size_threshold:

            # Larger image for dataset
            mX = X + (W // 2)
            mY = Y + (H // 2)
            R = int((1 + self.margin) * max(H,W) / 2)
            face = image[ max(0, mY-R) : mY+R, max(0, mX-R) : mX+R ]
            face = imresize(face, self.face_size, soft=True)

            # Closely cropped and aligned faces for feature extraction
            aligned_face = self.align(image, X, Y, W, H, landmarks)
            aligned_face = imresize(aligned_face, self.aligned_size)

            blur = detect_blur(aligned_face, image_size=None, center_crop=None)
            if blur > self.blur_threshold:

                detection['image'] = image_shape
                detection['face'] = face.shape
                detection['crop'] = aligned_face.shape

                return [detection], [face], [aligned_face]

        return [], [], []
        

    def align(self, image, X, Y, W, H, landmarks):
        """Crop and rotate a face to make eyes horizontal

        Args:
            image (ndarray): Image as a numpy array
            X (int): x-coord of top left of bounding box
            Y (int): y-coord of top left of bounding box
            W (int): Width of bounding box
            H (int): Height of bounding box
            landmarks (dict): Coordinates of facial landmarks

        Returns:
            ndarray: Cropped and aligned face as a numpy array
        """
        # Crop a larger portion of image to account for rotation
        pH, pW = int(0.3 * H), int(0.3 * W)
        dH1, dH2 = min(pH, Y), min(pH, image.shape[0] - Y+H) 
        dW1, dW2 = min(pW, X), min(pW, image.shape[1] - X+W)

        padded_crop = image[ Y-dH1 : Y+H+dH2, X-dW1 : X+W+dW2 ]

        leye = landmarks['left_eye']
        reye = landmarks['right_eye']

        # If dx(eyes) > dy(eyes), rotate image to make eyes horizontal
        # For small dx, rotation angle was too large
        if np.abs(reye[0] - leye[0]) > np.abs(reye[1] - leye[1]):        
            padded_crop = alignment_procedure(padded_crop, leye, reye)

        # Remove the extra portion added at the start
        face = padded_crop[dH1 : -dH2, dW1 : -dW2]

        return face


    def compare_with_previous(self, image, detections):

        res = 0
        if len(detections):
            if self.previous_image is not None:
                sim = template_matching(image, self.previous_image, (1280, 720))
                mean_iou = self.match_detections(detections, self.previous_dets)
                if sim > self.similarity_threshold and mean_iou > self.iou_threshold:
                    res = 1

            if self.previous_previous_image is not None and res == 0:
                sim = template_matching(image, self.previous_previous_image, (1280, 720))
                mean_iou = self.match_detections(detections, self.previous_previous_dets)
                if sim > self.similarity_threshold and mean_iou > self.iou_threshold:
                    res = 2

        self.previous_previous_image = self.previous_image
        self.previous_image = image
        self.previous_previous_dets = self.previous_dets
        self.previous_dets = detections

        return res


    def match_detections(self, det1, det2):

        score = 0.0
        if len(det1) == len(det2):
            boxes1 = self.extract_boxes(det1)
            boxes2 = self.extract_boxes(det2)
            score = mean_boxes_iou(boxes1, boxes2)
        
        return score


    def extract_boxes(self, dets):

        boxes = [
            np.array([
                det['box'][0],
                det['box'][1],
                det['box'][0] + det['box'][2],
                det['box'][1] + det['box'][3],
            ])
            for det in dets
        ]
        return np.array(boxes)


    def getmodel(self, method):

        if method == 'mtcnn':

            from .methods.mtcnn import MTCNNDetector

            model = MTCNNDetector()

        elif method == 'retinaface':

            from .methods.retinaface import RetinaFaceDetector

            model = RetinaFaceDetector()

        else:
            msg = f'Detector \'{method}\' is not available. '
            msg = msg + 'Use FaceDetection.available_detectors() to get a list of available detectors.'
            raise Exception(msg)

        return model