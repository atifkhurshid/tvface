import numpy as np
from PIL import Image
from deepface import DeepFace


class ArcfaceRepresentor(object):
    def __init__(self):
        self.model = DeepFace.build_model('ArcFace')
        self.model.predict(np.zeros(shape=(1, *self.model.input_shape[1:])))
        
        self.input_size = tuple(self.model.input_shape[1:3])  # (height, width)

    def represent(self, imgs):
        imgs_T = self.normalize(imgs)
        embs = self.model.predict(imgs_T)

        return embs

    def normalize(self, imgs):
        imgs_T = []
        for img in imgs:
            img_T = Image.fromarray(img)
            img_T = np.array(img_T.resize(self.input_size[::-1])) # PIL expects (width, height)
            imgs_T.append(img_T)

        imgs_T = np.array(imgs_T)
        imgs_T = imgs_T - 127.5
        imgs_T = imgs_T / 128.0

        return imgs_T
