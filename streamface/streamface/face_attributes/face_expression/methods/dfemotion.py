import numpy as np

from PIL import Image
from deepface import DeepFace


class DFEmotion(object):

    _emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self):

        self.model = DeepFace.build_model('Emotion')
        self.model.predict(np.zeros(shape=(1, *self.model.input_shape[1:])))
        
        self.input_size = tuple(self.model.input_shape[1:3])  # (height, width)


    def predict(self, faces):

        faces_T = self.normalize(faces)

        preds = self.model.predict(faces_T)

        res_list = []
        for pred in preds:
            res = self.process_prediction(pred)
            res_list.append(res)

        return res_list


    def process_prediction(self, pred):

        emotions = {}

        pred_sum = pred.sum()
        for i, emotion in enumerate(self._emotions):
            score = pred[i] / pred_sum
            emotions[emotion] = round(float(score), 2)

        return emotions


    def normalize(self, faces):

        faces_T = []
        for face in faces:
            face_T = Image.fromarray(face).convert('L')
            face_T = np.array(face_T.resize(self.input_size[::-1])) # PIL expects (width, height)
            face_T = face_T / 255.0
            face_T = np.expand_dims(face_T, axis=-1)
            faces_T.append(face_T)

        faces_T = np.array(faces_T)

        return faces_T
