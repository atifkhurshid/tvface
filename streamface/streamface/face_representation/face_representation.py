

class FaceRepresentation(object):
    _representations = ['arcface', 'facenet', 'arcfacenet']
    
    def __init__(self, method):
        self.model = self.getmodel(method)
    
    @classmethod
    def available_representations(cls):
        return cls._representations

    def getembeddings(self, faces):
        embs = self.model.represent(faces)
        return embs

    def getmodel(self, method):

        if method == 'arcface':

            from .methods.arcface import ArcfaceRepresentor

            model = ArcfaceRepresentor()

        elif method == 'facenet':

            from .methods.facenet import FacenetRepresentor

            model = FacenetRepresentor()

        elif method == 'arcfacenet':

            from .methods.arcfacenet import ArcFacenetRepresentor
            
            model = ArcFacenetRepresentor()

        else:
            msg = 'Representation \'{}\' is not available. '.format(method)
            msg = msg + 'Use FaceRepresentation.available_representations() to get a list of available representations.'
            raise Exception(msg)

        return model