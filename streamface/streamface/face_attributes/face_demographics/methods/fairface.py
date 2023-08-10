import gdown
import torch
import torchvision

import numpy as np
import torch.nn as nn

from torchvision import transforms
from pathlib import Path


class FairFace(object):
    """Implementation of FairFace face attribute classifier

    Modified from https://github.com/dchen236/FairFace
    """

    _weights_url = 'https://drive.google.com/uc?id=113QMzQzkBDmYMs9LwzvD-jxEZdBQ5J4X'

    _races = ('White', 'Black', 'Latino Hispanic', 'East Asian',
              'Southeast Asian', 'Indian', 'Middle Eastern')
    
    _genders = ('Male', 'Female')

    _ages = ('0-2', '3-9', '10-19', '20-29', '30-39', '40-49',
               '50-59', '60-69', '70+')

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = torch.device("cpu")

        self.trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.model = torchvision.models.resnet34(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 18)

        self.model.load_state_dict(torch.load(self.get_weights(), map_location=self.device))

        self.model = self.model.to(self.device)
        self.model.eval()


    def predict(self, faces):
        """Estimate race, gender and age from facial images

        Args:
            images (ndarray): Batch of images as a numpy array

        Returns:
            ndarray: Prediction scores as numpy array
        """
        faces_T = self.normalize(faces)
        faces_T = faces_T.to(self.device)

        preds = self.model(faces_T)
        preds = preds.cpu().detach().numpy()

        res_list = []
        for pred in preds:
            res = self.process_prediction(pred)
            res_list.append(res)

        return res_list


    def process_prediction(self, pred):
        
        race_pred = pred[:7]
        gender_pred = pred[7:9]
        age_pred = pred[9:18]

        age_score = np.exp(age_pred) / np.sum(np.exp(age_pred))
        gender_score = np.exp(gender_pred) / np.sum(np.exp(gender_pred))
        race_score = np.exp(race_pred) / np.sum(np.exp(race_pred))

        age_dict = {}
        age_sum = age_score.sum()
        for i, age in enumerate(self._ages):
            score = age_score[i] / age_sum
            age_dict[age] = round(float(score), 2)

        gender_dict = {}
        gender_sum = gender_score.sum()
        for i, gender in enumerate(self._genders):
            score = gender_score[i] / gender_sum
            gender_dict[gender] = round(float(score), 2)

        race_dict = {}
        race_sum = race_score.sum()
        for i, race in enumerate(self._races):
            score = race_score[i] / race_sum
            race_dict[race] = round(float(score), 2)

        res = {
            'age' : age_dict,
            'gender' : gender_dict,
            'race' : race_dict,
        }

        return res


    def normalize(self, faces):

        faces_T = []
        for face in faces:
            face_T = self.trans(face)
            faces_T.append(face_T)

        return torch.stack(faces_T)


    def get_weights(self):

        home = Path.home() / '.fairface' / 'weights'
        home.mkdir(parents=True, exist_ok=True)

        output = home / 'fairface_model_weights.pt'

        if not output.is_file():
            print("fairface_model_weights.pt will be downloaded...")

            gdown.download(self._weights_url, str(output), quiet=False)

        return str(output)
