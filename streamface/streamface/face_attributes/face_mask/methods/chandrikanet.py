"""
Face Mask Detection by chandrikadeb

Source: https://github.com/chandrikadeb7/Face-Mask-Detection
"""
import numpy as np
import tensorflow as tf

from pathlib import Path
from urllib.request import urlretrieve

from ....utils import imresize


class ChandrikaNet(object):

    def __init__(self):
        
        self.model = self.load_model()


    def predict(self, imgs):
        
        imgs_T = self.normalize(imgs)

        preds = self.model.predict(imgs_T)
        mask_probs = preds[:, 0]
        mask_probs = np.around(mask_probs.astype(float), 2)

        return mask_probs


    def normalize(self, imgs):

        imgs_T = []
        for img in imgs:

            img_T = imresize(img, size=(224, 224))
            imgs_T.append(img_T)
        
        imgs_T = np.array(imgs_T)

        imgs_T = tf.keras.applications.mobilenet_v2.preprocess_input(imgs_T)

        return imgs_T


    def load_model(self):

        home = Path.home() / '.chandrikanet' / 'weights'
        home.mkdir(parents=True, exist_ok=True)

        path = home / 'mask_detector.model'

        if not path.is_file():
            print("mask_detector.model will be downloaded...")

            url = 'https://github.com/chandrikadeb7/Face-Mask-Detection/raw/master/mask_detector.model'

            urlretrieve(url, path)

        model = tf.keras.models.load_model(str(path))

        return model
