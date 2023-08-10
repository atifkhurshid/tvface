import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm

from sklearn.metrics import silhouette_score

from ._face_clustering import BatchedHierarchicalClustering
from .cluster_merging import ClusterMergingAuto

from .utils import pkread, jwrite, oscopy


class FaceClustering(object):
    """Sort faces into groups using feature matching"""

    def __init__(self, features_path, matches_path, faces_dir, output_dir,
            metric, linkage, matching_threshold, matching_batchsize, merging_threshold):

        self.metric = metric

        self.paths = self.getpaths(
            features_path, matches_path, faces_dir, output_dir)

        self.model = BatchedHierarchicalClustering(
            metric, linkage, matching_threshold, matching_batchsize)

        self.merger = ClusterMergingAuto(metric, merging_threshold)


    def call(self):
        """Process face features for annotation"""

        print('Reading features file {} ... '.format(self.paths['features']))
        features = pkread(self.paths['features'])

        print('Reading matches file {} ... '.format(self.paths['matches']))
        df = pd.read_csv(self.paths['matches'])
        matches = pd.Series(df['Face 1'].values, index=df['Face 2']).to_dict()

        print('Initializing Face Clustering ...')
        self.cluster(features, matches)


    def cluster(self, features, matches):
        """Sort faces into groups based on feature matching

        Stats are added to a dictionary and periodically written to a csv file.
        Face images are copied to a new directory based on face labels.

        Args:
            filepath (pathlib.Path): Path of feature file
            n (int): Iteration number
        """
        filenames = list(features.keys())
        embeddings, attributes, expressions, maskprobs, poses = zip(*list(features.values()))

        print(f'Clustering {len(filenames)} samples ... ')
        labels = self.model.cluster(embeddings)

        print(f'Merging {len(set(labels))} clusters ...')
        labels = self.merger.mergelabels(
            filenames, embeddings, labels, matches)

        print(f'Arranging {len(set(labels))} clusters ...')
        labels = self.order_labels_by_frequency(labels)

        print('Calculating Silhouette score: ', end='', flush=True)
        score = silhouette_score(embeddings, labels, metric=self.metric)
        print('{:.3f}'.format(score))

        print('Writing {} annotations to {} ... '.format(
            len(filenames), self.paths['annotations']))
        self.writeannotations(
            filenames, labels, attributes, expressions, maskprobs, poses)

        self.copyfaces(filenames, labels)

        print('Closing Face Clustering ...')


    def order_labels_by_frequency(self, labels):

        unique_elements, frequency = np.unique(labels, return_counts=True)
        sorted_indexes = np.argsort(frequency)[::-1]
        sorted_by_freq = unique_elements[sorted_indexes]

        sorted_labels = labels.copy()
        for i, y in enumerate(sorted_by_freq):
            sorted_labels[labels == y] = i

        return sorted_labels


    def writeannotations(
            self, filenames, labels, attributes, expressions, maskprobs, poses):
        """Write annotations to json files

        Path: output_dir / metadata
        Filename: annotations.json
        Format:

            'labels' : {
                'file 1' : {
                    'label' : 0,
                    'mask' : 0.1,
                    "attributes": {
                        "age": {
                            "0-2": 0.0,
                            "3-9": 0.0,
                            "10-19": 0.0,
                            "20-29": 0.0,
                            "30-39": 0.18,
                            "40-49": 0.76,
                            "50-59": 0.05,
                            "60-69": 0.0,
                            "70+": 0.0
                        },
                        "gender": {
                            "Male": 1.0,
                            "Female": 0.0
                        },
                        "race": {
                            "White": 0.8,
                            "Black": 0.0,
                            "Latino Hispanic": 0.09,
                            "East Asian": 0.0,
                            "Southeast Asian": 0.0,
                            "Indian": 0.01,
                            "Middle Eastern": 0.09
                        },
                        "expression": {
                            "angry": 0.49,
                            "disgust": 0.0,
                            "fear": 0.36,
                            "happy": 0.0,
                            "sad": 0.14,
                            "surprise": 0.0,
                            "neutral": 0.01
                        },
                        "pose": {
                            "yaw": -18.84,
                            "pitch": -1.47,
                            "roll": 0.92
                        }
                    }
                }
            }
        """
        annotations_dict = {}

        for filename, label, attribute, expression, maskprob, pose in zip(
                filenames, labels, attributes, expressions, maskprobs, poses):

            annotation = {
                'label' : int(label),
                'mask' : maskprob,
                'attributes' : {
                    **attribute,
                    'expression' : {
                        **expression,
                    },
                    'pose' : {
                        **pose,
                    },
                },
            }
            annotations_dict[Path(filename).stem] = annotation

        annotations_dict = {
            'labels' : annotations_dict
        }

        jwrite(self.paths['annotations'], annotations_dict)


    def copyfaces(self, filenames, labels):

        x = input('Copy face images to labeled folder? [Y/N] : ')
        if x.lower() == 'y':

            print('Copying {} faces to {} ... '.format(
                len(filenames), self.paths['labeled_faces']))

            for filename, label in tqdm(
                    zip(filenames, labels), total=len(filenames)):

                src_facepath = self.paths['faces'] / filename
                tgt_facepath = self.paths['labeled_faces'] / str(label)
                oscopy(str(src_facepath), str(tgt_facepath))

        else:
            print('Face images not copied.')


    def getpaths(self, features_path, matches_path, faces_dir, output_dir):
        """Define filepaths for reading faces and features,
        and writing faces, annotations, and logs

        Create filepaths if they do not exist.

        Args:
            features_path (str): Path of pickled features
            faces_dir (str): Directory containing faces and aligned_aligned
            output_dir (str): Output directory

        Returns:
            dict: Dictionary of pathlib.Path objects
        """
        paths = {
            'faces' : Path(faces_dir),
            'output' : Path(output_dir),
            'labeled_faces' : Path(output_dir) / 'labeled_faces',
            'metadata' : Path(output_dir) / 'metadata',
        }
        for _, path in paths.items():
            path.mkdir(exist_ok=True, parents=True)

        paths['annotations'] = paths['metadata'] / 'annotations.json'
        
        paths['features'] = Path(features_path)
        paths['matches'] = Path(matches_path)

        return paths
