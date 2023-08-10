import time
import shutil
import traceback

from datetime import datetime
from pathlib import Path

from .utils import Logging
from .utils import DirectoryReader
from .utils import DelayedKeyboardInterrupt
from .utils import imread, pkwrite, pkmerge

from .face_representation import FaceRepresentation
from .face_attributes import FaceDemographics
from .face_attributes import FaceExpression
from .face_attributes import FaceMask
from .face_attributes import FacePose


class FaceAnalysis(object):
    """Exract features and attributes from faces"""

    def __init__(
            self,
            input_dir,
            output_dir,
            representation='arcfacenet',
            demographics='fairface',
            expression='dfemotion',
            mask='chandrikanet',
            pose='whenet',
            batch_size=128,
            resume=True,
            log_interval=100
        ):

        self.batch_size = batch_size
        self.log_interval = log_interval

        self.paths = self.getpaths(input_dir, output_dir)
        self.logger = self.getlogger()

        self.faces_dir = DirectoryReader(
            self.paths['aligned_faces'], ['.png', '.jpg'], self.logger, resume)
        
        self.representor = FaceRepresentation(representation)
        self.demographor = FaceDemographics(demographics)
        self.expressor = FaceExpression(expression)
        self.masker = FaceMask(mask)
        self.poser = FacePose(pose)


    def call(self):
        """Process face images in an infinite loop
        
        In each loop:
            - Get paths of face images in input directory
            - Extract features and attributes from faces
        Use KeyboardInterrupt (Ctrl + C) to end loop.
        """
        try:
            self.logger.log('Initializing Face Analysis', 'INFO')
            n = -1
            while True:
                n += 1
                try:
                    # =========================================================== #

                    nofiles = False
                    with DelayedKeyboardInterrupt():
                        filepaths = self.faces_dir.next(self.batch_size)
                        if len(filepaths):
                            self.analyze(filepaths)
                        else:
                            nofiles = True

                    if nofiles:
                        msg = 'Could not find any new files. Retrying in 30 minutes...'
                        self.logger.log(msg, 'ERR')

                        time.sleep(30 * 60)

                        continue

                    # =========================================================== #

                    if n  % self.log_interval == 0:
                        msg = f'Processing - Iteration {n} - File {filepaths[0].name}'
                        self.logger.log(msg, 'INFO')
                        
                except KeyboardInterrupt:
                    msg = 'Close instruction received. Wrapping up final iteration'
                    self.logger.log(msg, 'INFO')
                    break

                except Exception:
                    msg = f'Exception - Iteration {n} - File {filepaths[0].name}'
                    self.logger.log(msg, 'ERR')
                    self.logger.log(traceback.format_exc(), 'ERR')
                    continue
        finally:
            self.close()

    # =========================================================== #

    def analyze(self, filepaths):
        """Extract feature vectors and attributes from face images

        Args:
            filepath (pathlib.Path): Path of file containing face paths
            n (int): Iteration number
        """
        faces_dict, cropped_faces_dict, aligned_faces_dict = self.readfaces(filepaths)
        filenames = list(faces_dict.keys())

        faces = list(faces_dict.values())
        poses = self.poser.getposes(faces)

        cropped_faces = list(cropped_faces_dict.values())
        attributes = self.demographor.getdemographics(cropped_faces)
        expressions = self.expressor.getexpressions(cropped_faces)
        maskprobs = self.masker.getmaskprobs(cropped_faces)

        aligned_faces = list(aligned_faces_dict.values())
        embeddings = self.representor.getembeddings(aligned_faces)


        self.savefeatures(
            filenames, embeddings, attributes, expressions, maskprobs, poses)

    # =========================================================== #


    def readfaces(self, filepaths):
        """Read face image files from predefined directories

        Paths:
            - input_dir / faces
            - input_dir / aligned_faces

        Args:
            filepaths (list): List of pathlib.Path objects

        Returns:
            dict: Dictionary of images in numpy arrays, key = filename
        """
        faces = {}
        cropped_faces = {}
        aligned_faces = {}
        for filepath in filepaths:
            filename = filepath.name
            try:
                facepath = self.paths['faces'] / filename
                aligned_facepath = self.paths['aligned_faces'] / filename

                face = imread(facepath)
                aligned_face = imread(aligned_facepath)

                p = int(0.15 * face.shape[0])
                faces[filename] = face[p:-p, p:-p]

                p = int(0.25 * face.shape[0])
                cropped_faces[filename] = face[p:-p, p:-p]

                aligned_faces[filename] = aligned_face

            except KeyboardInterrupt:
                raise
            except:
                continue

        return faces, cropped_faces, aligned_faces


    def savefeatures(
            self, filenames, embeddings, attributes, expressions, maskprobs, poses):
        """Save face features in pickle files

            - Path: output_dir / features
            - Filenames: {name of first face file in batch}.pkl

        Args:
            filenames (list): List of names of face image files
            embeddings (list): List of embedding of respective faces
            attributes_list (list): List of attribute dictionaries of respective faces
        """
        data = dict(
            zip(
                filenames,
                zip(
                    embeddings,
                    attributes,
                    expressions,
                    maskprobs,
                    poses
                )
            )
        )
        featurepath = self.paths['temp_features'] / f'{Path(filenames[0]).stem}.pkl'
        pkwrite(featurepath, data)


    def mergefeatures(self):
        features = pkmerge(self.paths['temp_features'])

        msg = 'Merging {} features and saving at {}'.format(len(features), self.paths['features'])
        self.logger.log(msg, 'SYS')

        first_name = list(features.keys())[0]
        filename = f'{Path(first_name).stem}.pkl'
        pkwrite(self.paths['features'] / filename, features)

        msg = 'Deleting temp features from {}'.format(self.paths['temp_features'])
        self.logger.log(msg, 'SYS')

        shutil.rmtree(self.paths['temp_features'])


    def getpaths(self, input_dir, output_dir):
        """Define filepaths for reading faces and
        writing features, attributes, and logs

        Create filepaths if they do not exist.

        Args:
            input_dir (str): Input directory
            output_dir (str): Output directory

        Returns:
            dict: Dictionary of pathlib.Path objects
        """
        paths = {
            'input' : Path(input_dir),
            'faces' : Path(input_dir) / 'faces',
            'aligned_faces' : Path(input_dir) / 'aligned_faces',
            'output' : Path(output_dir),
            'temp_features' : Path(output_dir) / 'temp_features',
            'features' : Path(output_dir) / 'features',
            'logs' : Path(output_dir) / 'logs',
        }
        for _, path in paths.items():
            path.mkdir(exist_ok=True, parents=True)

        return paths


    def getlogger(self):
        """Initialize logging object

        Path: output_dir/logs
        Filename: face_analysis_log_timestamp.txt
        """
        name = 'face_analysis_log_{:%Y%m%d%H%M}.txt'.format(datetime.now())
        filepath = self.paths['logs'] / name
        logger = Logging(filepath)

        return logger


    def close(self):
        """Store data and matcher, release resources on exit

        Always executed when process exits.
        """
        x = input('Merge features? [Y/N] : ')
        if x.lower() == 'y':
            self.mergefeatures()
        else:
            print('Merging cancelled.')

        self.logger.log('Closing Face Analysis', 'INFO')
        self.logger.close()
