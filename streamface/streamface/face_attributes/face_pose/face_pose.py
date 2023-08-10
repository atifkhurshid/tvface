

class FacePose(object):

    _models = ['whenet']

    def __init__(self, method):
        self.model = self.getmodel(method)

    @classmethod
    def available_models(cls):
        return cls._models


    def getposes(self, faces):
        """Determine face pose from images

        Uses the WHENet model to regress yaw, pitch and roll angles

        Args:
            faces (ndarray): Batch of images as a numpy array

        Returns:
            list of dicts: Pose dictionaries, key = pose angle name
        """
        pose_list = self.model.predict(faces)
        
        return pose_list


    def getmodel(self, method):

        if method == 'whenet':

            from .methods.whenet import WHENet

            model = WHENet()
            
        else:
            msg = 'Model \'{}\' is not available. '.format(method)
            msg = msg + 'Use FacePose.available_model() to get a list of available models.'
            raise Exception(msg)

        return model
