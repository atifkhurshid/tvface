

class FaceDemographics(object):

    _models = ['fairface']

    def __init__(self, method):
        self.model = self.getmodel(method)

    @classmethod
    def available_models(cls):
        return cls._models


    def getdemographics(self, faces):
        """Determine facial demographics from images

        Uses the FairFace model to jointly classify age, race, and gender
        from facial images.

        Args:
            images (ndarray): Batch of images as a numpy array

        Returns:
            list of dicts: Attribute dictionaries, key = attribute name
        """
        demographics_list = self.model.predict(faces)

        return demographics_list


    def getmodel(self, method):

        if method == 'fairface':

            from .methods.fairface import FairFace
    
            model = FairFace()

        else:
            msg = 'Model \'{}\' is not available. '.format(method)
            msg = msg + 'Use FaceDemographics.available_model() to get a list of available models.'
            raise Exception(msg)

        return model
