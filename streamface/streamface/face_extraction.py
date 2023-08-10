import time
import traceback

import pandas as pd

from pathlib import Path
from datetime import datetime

from .utils import Logging
from .utils import DirectoryReader
from .utils import DelayedKeyboardInterrupt
from .utils import imread, imwrite

from .face_detection import FaceDetection


class FaceExtraction(object):
    """Extract faces from images of video frames."""

    def __init__(
            self,
            input_dir,
            output_dir,
            detection='retinaface',
            batch_size=32,
            conf_threshold=0.95,
            size_threshold=0.005,
            blur_threshold=25,
            match_thresholds=(0.75, 0.75),
            face_size=(256, 256),
            aligned_size=(128, 128),
            padding=0.5,
            margin=1.0,
            resume=True,
            log_interval=100
        ):
        
        self.batch_size = batch_size
        self.log_interval = log_interval

        self.paths = self.getpaths(input_dir, output_dir, resume)
        self.logger = self.getlogger()

        self.frames_dir = DirectoryReader(
            self.paths['frames'], ['.png', '.jpg'], self.logger, resume)
        
        self.detector = FaceDetection(
            detection, conf_threshold, size_threshold, blur_threshold,
            match_thresholds, face_size, aligned_size, padding, margin)

        self.previous_filename = None
        self.previous_previous_filename = None


    def call(self):
        """Process frame images in an infinite loop
        
        In each loop:
            - Get path of next frame image in input directory
            - Perform face detection and extraction
        Use KeyboardInterrupt (Ctrl + C) to end loop.
        """
        try:
            self.logger.log('Initializing Face Extraction', 'INFO')
            n = -1
            while True:
                n += 1
                try:
                    # =========================================================== #

                    nofiles = False
                    with DelayedKeyboardInterrupt():
                        filepaths = self.frames_dir.next(self.batch_size)
                        if len(filepaths):
                            self.extract(filepaths)
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

    def extract(self, filepaths):
        """Detect and save faces alongside bounding boxes + landmarks

        Face detector returns bounding box info, loosely cropped face
        images for storage alongside closely cropped and aligned images
        for embedding and attribute extraction.
        
        Args:
            filepath (pathlib.Path): Path of frame image file
            n (int): Iteration number
        """
        faces_dict = {}
        aligned_faces_dict = {}
        detections_dict = {}
        matches_dict = {}

        frames = self.readframes(filepaths)
        for filename, frame in frames.items():

            dets, faces, aligned_faces, match = self.detector.getfaces(frame)

            for i, (det, face, aligned_face) in enumerate(zip(dets, faces, aligned_faces)):
                
                facename = self.getfacefilename(filename, i)

                detections_dict[facename] = det
                faces_dict[facename] = face
                aligned_faces_dict[facename] = aligned_face

            processed_matches = self.processmatches(match, filename, len(faces))
            if len(processed_matches):
                matches_dict = {**matches_dict, **processed_matches}

        self.savefaces(faces_dict, aligned_faces_dict)
        self.savedetections(detections_dict)
        self.savematches(matches_dict)

    # =========================================================== #


    def readframes(self, filepaths):
        """Read a batch of frames from filepaths

        Args:
            filepaths (list): List of pathlib.Path objects

        Returns:
            list: List of images as numpy arrays
        """
        frames = {}
        for path in filepaths:
            try:
                image = imread(path)
                frames[path.name] = image
            except KeyboardInterrupt:
                raise
            except:
                continue
        
        return frames


    def getfacefilename(self, name, i):
        """Get the name of face image file in a predefined format

        Format: {name of frame image}_face_{number of face in frame}

        Args:
            name (str): Name of frame image file
            i (int): Count of faces in one frame

        Returns:
            str: Name of face image file to be written
        """
        path = Path(name)
        filename = f'{path.stem}_face_{i}{path.suffix}'

        return filename


    def processmatches(self, match, filename, num_faces):
        matches = {}
        if match is 1: # Match found in previous_filename
            for i in range(num_faces):
                current_facename = self.getfacefilename(filename, i)
                previous_facename = self.getfacefilename(self.previous_filename, i)
                matches[current_facename] = previous_facename

        elif match is 2: # Match found in previous_previous_filename
            for i in range(num_faces):
                current_facename = self.getfacefilename(filename, i)
                previous_previous_facename = self.getfacefilename(self.previous_previous_filename, i)
                matches[current_facename] = previous_previous_facename

        self.previous_previous_filename = self.previous_filename
        self.previous_filename = filename

        return matches


    def savefaces(self, faces, aligned_faces):
        """Save full and cropped face images to pre-defined paths

        Paths:
            - Faces: output_dir / faces
            - Aligned Faces: output_dir / aligned_faces

        Filenames:
            {name of frame image}_face_{number of face in frame}.png
        """
        for facename in faces:
            facepath = self.paths['faces'] / facename
            aligned_facepath = self.paths['aligned_faces'] / facename

            face = faces[facename]
            aligned_face = aligned_faces[facename]

            imwrite(facepath, face)
            imwrite(aligned_facepath, aligned_face)


    def savedetections(self, detections):
        """Save bounding boxes and landmarks in a pkl file

            - Path: output_dir / tempdetections
            - Filenames: facedetections_{name of first face file in batch}.pkl
        """
        if not self.paths['detections'].exists():
            columns = ['name', 'confidence', 'box', 'left_eye', 'right_eye', 'nose',
                       'mouth_left', 'mouth_right', 'image', 'face', 'crop',]
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.paths['detections'], index=False)

        if detections:
            data = []
            for name, values in detections.items():
                data.append([
                    name, values['confidence'], values['box'], *list(values['landmarks'].values()),
                    values['image'], values['face'], values['crop']
                ])
            df = pd.DataFrame(data)
            df.to_csv(self.paths['detections'], mode='a', index=False, header=False)

            msg = 'Saving {} detections at {}'.format(len(detections), self.paths['detections'])
            self.logger.log(msg, 'SYS')


    def savematches(self, matches):
        if not self.paths['matches'].exists():
            columns = ['Face 1', 'Face 2']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.paths['matches'], index=False)

        if matches:
            data = [(k, v) for k, v in matches.items()]
            df = pd.DataFrame(data)
            df.to_csv(self.paths['matches'], mode='a', header=False, index=False)

            msg = 'Saving {} matches at {}'.format(len(matches), self.paths['matches'])
            self.logger.log(msg, 'SYS')


    def getpaths(self, input_dir, output_dir, resume):
        """Define filepaths for reading frames and writing faces, features, and logs

        Create filepaths if they do not exist.

        Args:
            input_dir (str): Input directory
            output_dir (str): Output directory

        Returns:
            dict: Dictionary of pathlib.Path objects
        """
        paths = {
            'input' : Path(input_dir),
            'frames' : Path(input_dir) / 'frames',
            'output' : Path(output_dir),
            'faces' : Path(output_dir) / 'faces',
            'aligned_faces' : Path(output_dir) / 'aligned_faces',
            'metadata': Path(output_dir) / 'metadata',
            'logs' : Path(output_dir) / 'logs',
        }
        for _, path in paths.items():
            path.mkdir(exist_ok=True, parents=True)

        paths['detections'] = paths['metadata'] / 'detections.csv'
        paths['matches'] = paths['metadata'] / 'matches.csv'

        if not resume:
            # Delete old stats
            if paths['detections'].exists():
                paths['detections'].unlink()
            if paths['matches'].exists():
                paths['matches'].unlink()

        return paths


    def getlogger(self):
        """Initialize logging object

        Path: output_dir/logs
        Filename: face_extraction_log_timestamp.txt
        """
        name = 'face_extraction_log_{:%Y%m%d%H%M%S}.txt'.format(datetime.now())
        filepath = self.paths['logs'] / name
        logger = Logging(filepath)

        return logger
    

    def close(self):
        """Store data and release resources on exit

        Always executed when process exits.
        """
        self.logger.log('Closing Face Extraction', 'INFO')
        self.frames_dir.close()
        self.logger.close()
