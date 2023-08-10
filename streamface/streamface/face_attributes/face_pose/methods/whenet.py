"""
WHENet: Real-time Fine-Grained Estimation for Wide Range Head Pose

Source: https://github.com/Ascend-Research/HeadPoseEstimation-WHENet
"""

import numpy as np
import tensorflow as tf

from pathlib import Path
from urllib.request import urlretrieve
from efficientnet.tfkeras import EfficientNetB0

from ....utils import imresize, center_crop


class WHENet:
    def __init__(self):

        base_model = EfficientNetB0(
            include_top=False, input_shape=(224, 224, 3))

        out = base_model.output
        out = tf.keras.layers.GlobalAveragePooling2D()(out)

        fc_yaw = tf.keras.layers.Dense(name='yaw_new', units=120)(out) # 3 * 120 = 360 degrees in yaw
        fc_pitch = tf.keras.layers.Dense(name='pitch_new', units=66)(out)
        fc_roll = tf.keras.layers.Dense(name='roll_new', units=66)(out)
        
        self.model = tf.keras.models.Model(
            inputs=base_model.input, outputs=[fc_yaw, fc_pitch, fc_roll])
        
        self.load_weights()

        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = np.array(self.idx_tensor, dtype=np.float32)
        self.idx_tensor_yaw = [idx for idx in range(120)]
        self.idx_tensor_yaw = np.array(self.idx_tensor_yaw, dtype=np.float32)


    def predict(self, imgs):

        imgs_T = self.normalize(imgs)

        predictions = self.model.predict(imgs_T, batch_size=len(imgs_T))

        yaw_predicted = self.softmax(predictions[0])
        pitch_predicted = self.softmax(predictions[1])
        roll_predicted = self.softmax(predictions[2])

        yaw_predicted = np.sum(yaw_predicted*self.idx_tensor_yaw, axis=1) * 3 - 180
        pitch_predicted = np.sum(pitch_predicted * self.idx_tensor, axis=1) * 3 - 99
        roll_predicted = np.sum(roll_predicted * self.idx_tensor, axis=1) * 3 - 99

        pose_list = []
        for yaw, pitch, roll in zip(yaw_predicted, pitch_predicted, roll_predicted):
            pose = self.process_prediction(yaw, pitch, roll)
            pose_list.append(pose)

        return pose_list


    def normalize(self, imgs):

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        imgs_T = []
        for img in imgs:

            img_T = np.array(img)

            img_T = center_crop(img_T, margin=0.2)
            img_T = imresize(img_T, size=(224, 224))

            img_T = img_T / 255
            img_T = (img_T - mean) / std

            imgs_T.append(img_T)

        return np.array(imgs_T)


    def process_prediction(self, yaw, pitch, roll):
        
        pose = {
            'yaw' : round(float(yaw), 2),
            'pitch' : round(float(pitch), 2),
            'roll' : round(float(roll), 2),
        }

        return pose


    def softmax(self, x):
        
        x -= np.max(x,axis=1, keepdims=True)
        a = np.exp(x)
        b = np.sum(np.exp(x), axis=1, keepdims=True)

        return a/b


    def load_weights(self):

        home = Path.home() / '.whenet' / 'weights'
        home.mkdir(parents=True, exist_ok=True)

        path = home / 'WHENet.h5'

        if not path.is_file():
            print("WHENet.h5 will be downloaded...")

            url = 'https://github.com/Ascend-Research/HeadPoseEstimation-WHENet/raw/master/WHENet.h5'

            urlretrieve(url, path)

        self.model.load_weights(str(path))
