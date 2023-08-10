import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm
from pathlib import Path
from tensorflow.keras.optimizers import Adam
from deepface.basemodels.ArcFace import ResNet34
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .utils import L2Normalization, ArcLayer, SparseArcLoss

tf.keras.utils.set_random_seed(42)


class ArcFace(object):

    def __init__(self, shape=(256, 256, 3), n_classes=1000, regularizer=None):

        self.input_shape = shape
        self.n_classes = n_classes

        base_model = ResNet34()
        inputs = base_model.inputs[0]
        arcface_model = base_model.outputs[0]
        arcface_model = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5)(arcface_model)
        arcface_model = tf.keras.layers.Dropout(0.4)(arcface_model)
        arcface_model = tf.keras.layers.Flatten()(arcface_model)
        arcface_model = tf.keras.layers.Dense(512, activation=None, use_bias=True, kernel_initializer="glorot_normal")(arcface_model)
        embedding = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=2e-5, name="embedding", scale=True)(arcface_model)
        self.base_model = tf.keras.models.Model(inputs, embedding, name=base_model.name)

        arcface_weights_path = Path.home() / '.deepface' / 'weights' / 'arcface_weights.h5'
        # self.base_model.load_weights(arcface_weights_path)

        inputs2 = tf.keras.layers.Input(shape=shape)
        x = tf.keras.layers.CenterCrop(112, 112)(inputs2)
        x = self.base_model(x)
        # """
        x = L2Normalization()(x)
        x = ArcLayer(n_classes, regularizer)(x)
        """
        x = tf.keras.layers.Dense(n_classes, activation=None)(x)
        # """
        self.model = tf.keras.Model(inputs=inputs2, outputs=x)

        # self.model.load_weights('D:\StreamFace\data\model\checkpoint')

    @staticmethod
    def _preprocessing_fn(img):
        img = img - 127.5
        img = img / 128.0
        return img


    def train(self, df, faces_path, validation_split, batch_size,
            learning_rate, epochs, verbose=1):
        
        train_generator, val_generator = self._load_train_data(
            df, faces_path, validation_split, batch_size)

        H = self._fit(
            train_generator, val_generator, learning_rate, epochs, verbose)

        return H


    def test(self, df, faces_path, batch_size):

        test_generator = self._load_test_data(
            df, faces_path, batch_size)

        preds = self._predict(test_generator)
        y_pred = np.argmax(preds, axis=1)

        y_true = test_generator.labels

        scores = {
            'Accuracy' : 100 * accuracy_score(y_true, y_pred),
            'Precision' : 100 * precision_score(y_true, y_pred, average='macro'),
            'Recall' : 100 * recall_score(y_true, y_pred, average='macro'),
            'F1 Score' : 100 * f1_score(y_true, y_pred, average='macro'),
        }

        print('Classification Report:')
        for k, v in scores.items():
            print('\t{} = {:.2f}'.format(k, v))


    def annotate(self, df, faces_path, annotations_path, batch_size,
            threshold):

        data_generator = self._load_test_data(
            df, faces_path, batch_size)
        y_true = data_generator.labels

        preds = self._predict(data_generator)

        preds = preds / preds.max()

        y_pred = []
        for pred in preds:
            idx = np.argmax(pred)
            # If low confidence or last class, label as noisy (Last class is noisy)
            if pred[idx] < threshold or idx == len(pred) - 1:
                y_pred.append(-1)
            else:
                y_pred.append(idx)
                
        y_pred = np.array(y_pred)
        
        if annotations_path is not None:
            self._write_annotations(
                annotations_path, data_generator.filenames, y_true, y_pred)


    def save(self, prediction_path=None, embedding_path=None):

        if prediction_path is not None:
            Path(prediction_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(prediction_path)
        
        if embedding_path is not None:
            Path(embedding_path).parent.mkdir(parents=True, exist_ok=True)
            self.base_model.save_weights(embedding_path)


    def load(self, prediction_path=None, embedding_path=None):

        if prediction_path is not None:
            self.model.load_weights(prediction_path)
        
        if embedding_path is not None:
            self.base_model.load_weights(embedding_path)


    def _fit(self, training_data, validation_data, learning_rate, epochs, verbose):
        # """
        def scheduler(epoch, lr):
            if epoch == 10:
                return lr / 100
            if epoch == 20:
                return lr / 10
            if epoch == 30:
                return lr
            if epoch == 40:
                return lr 
            return lr
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
        lr_callback2 = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1,
            mode='auto',
            min_delta=0.1,
            cooldown=1
        )

        log_dir = "./tensorboard/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            "D:\StreamFace\data\model\checkpoint",
            monitor='val_loss',
            verbose=1,
            save_best_only = True,
            save_weights_only = True,
            mode = 'min',
            save_freq='epoch',
        )
        # """
        """
        self.base_model.trainable = False

        self.model.compile(
            optimizer=Adam(learning_rate=0.1),
            loss=SparseArcLoss(self.n_classes),
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        H = self.model.fit(
            training_data,
            epochs=1,
            validation_data=validation_data,
            verbose=verbose,
        )
        # """
        # self.base_model.trainable = True

        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=SparseArcLoss(self.n_classes),
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        H = self.model.fit(
            training_data,
            epochs=epochs,
            validation_data=validation_data,
            verbose=verbose,
            # callbacks=[lr_callback, tensorboard_callback, checkpoint_callback]
        )
        # self.base_model.trainable = True

        return H


    def _predict(self, data):

        preds = []
        for i in range(len(data)):
            x, _ = data[i]
            pred = self.model(x, training=False)
            pred = tf.nn.softmax(pred, axis=-1).numpy()
            preds.extend(pred)
        preds = np.array(preds)

        return preds


    def _load_train_data(self, df, faces_path, validation_split, batch_size):

        '''
        df = df.groupby(
                'class', group_keys=False
            ).apply(
                lambda x: x.sample(50, replace=True)
            )
        #'''

        df = df.sample(frac=1)  # Shuffling data

        datagen = ImageDataGenerator(
            preprocessing_function=ArcFace._preprocessing_fn,
            validation_split=validation_split
        )

        train_generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=faces_path,
            target_size=self.input_shape[:-1],
            color_mode='rgb',
            class_mode='raw',
            batch_size=batch_size,
            shuffle=True,
            subset='training',
            validate_filenames=True,
        )

        val_generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=faces_path,
            target_size=self.input_shape[:-1],
            color_mode='rgb',
            class_mode='raw',
            batch_size=batch_size,
            shuffle=True,
            subset='validation',
            validate_filenames=True,
        )

        return train_generator, val_generator


    def _load_test_data(self, df, faces_path, batch_size):

        datagen = ImageDataGenerator(
            preprocessing_function=ArcFace._preprocessing_fn
        )

        test_generator = datagen.flow_from_dataframe(
            dataframe=df,
            directory=faces_path,
            target_size=self.input_shape[:-1],
            color_mode='rgb',
            class_mode='raw',
            batch_size=batch_size,
            shuffle=False,
            validate_filenames=True,
        )

        return test_generator


    def _write_annotations(self, annotations_path, filenames, y_trues, y_preds):
    
        annotations_dict = {}

        for filename, y_true, y_pred in zip(
                filenames, y_trues, y_preds):

            annotations_dict[Path(filename).stem] = {
                'label' : int(y_pred),
                'true' : int(y_true),
            }

        annotations_dict = {
            'labels' : annotations_dict
        }

        with open(annotations_path, 'w', encoding='utf-8') as fp:
            json.dump(annotations_dict, fp, ensure_ascii=False, indent=4)
