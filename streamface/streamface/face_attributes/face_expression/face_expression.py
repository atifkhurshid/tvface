

class FaceExpression(object):

    _models = ['dfemotion']

    def __init__(self, method):
        self.model = self.getmodel(method)

    @classmethod
    def available_models(cls):
        return cls._models


    def getexpressions(self, faces):
        """Determine facial expressions from images

        Uses the DeepFace Emotion model to classify emotion
        from facial images.

        Args:
            images (ndarray): Batch of images as a numpy array

        Returns:
            list of dicts: Emotion dictionaries, key = emotion name
        """
        expressions_list = self.model.predict(faces)

        return expressions_list


    def getmodel(self, method):

        if method == 'dfemotion':

            from .methods.dfemotion import DFEmotion

            model = DFEmotion()
            
        else:
            msg = 'Model \'{}\' is not available. '.format(method)
            msg = msg + 'Use FaceExpression.available_model() to get a list of available models.'
            raise Exception(msg)

        return model
