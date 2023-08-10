import numpy as np

from sklearn.preprocessing import normalize

from .arcface import ArcfaceRepresentor
from .facenet import FacenetRepresentor


class ArcFacenetRepresentor(object):
    def __init__(self):
        self.arcface = ArcfaceRepresentor()
        self.facenet = FacenetRepresentor()

    def represent(self, imgs):
        arc_embs = self.arcface.represent(imgs)
        arc_embs = normalize(arc_embs, norm='l2', axis=1, copy=False, return_norm=False)
        
        face_embs = self.facenet.represent(imgs)
        face_embs = normalize(face_embs, norm='l2', axis=1, copy=False, return_norm=False)
        
        embs = np.hstack((arc_embs, face_embs))
        # embs = arc_embs + face_embs
        embs = normalize(embs, norm='l2', axis=1, copy=False, return_norm=False)

        return embs

