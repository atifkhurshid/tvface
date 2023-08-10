

class FaceMask(object):

    _models = ['chandrikanet']

    def __init__(self, method):
        self.model = self.getmodel(method)

    @classmethod
    def available_models(cls):
        return cls._models


    def getmaskprobs(self, faces):
        """Detect face masks in images

        Uses the ChandrikaNet model to classify mask probability

        Args:
            faces (ndarray): Batch of images as a numpy array

        Returns:
            np.array: Mask probabilities
        """
        mask_probs = self.model.predict(faces)

        return mask_probs


    def getmodel(self, method):

        if method == 'chandrikanet':

            from .methods.chandrikanet import ChandrikaNet

            model = ChandrikaNet()
            
        else:
            msg = 'Model \'{}\' is not available. '.format(method)
            msg = msg + 'Use FaceMask.available_model() to get a list of available models.'
            raise Exception(msg)

        return model
