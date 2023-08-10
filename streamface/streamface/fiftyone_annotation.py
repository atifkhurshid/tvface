import pandas as pd
import fiftyone as fo

from pathlib import Path
from fiftyone import ViewField as F

from .cluster_merging import ClusterMergingManual

from .utils import jread, jwrite


class FiftyOneAnnotation(object):
    """Manual annotation using FiftyOne"""

    def __init__(
            self, name, faces_path, evals_path,
            annotations_path, new_annotations_path):

        self.name = name
        self.faces_path = faces_path
        self.evals_path = evals_path
        self.annotations_path = annotations_path
        self.new_annotations_path = new_annotations_path

        self.merger = ClusterMergingManual()

        self.matches_path = Path('./temp_matches.csv')
        self.ignore_path = Path('./temp_ignore.csv')
        self.skip_path = Path('./temp_skip.csv')
        self.noisy_path = Path('./temp_noisy.csv')


    def annotate(self):

        print('Initializing FiftyOne Annotation ...')

        print('Loading matches ...')
        matches = jread(self.evals_path)['matches']
        if not len(matches):
            print('No matched found.')
            return

        print('Loading dataset ...')
        dataset = self.load_dataset()
        annotations = jread(self.annotations_path)

        print('Launching fiftyone app ...')
        config = fo.core.config.AppConfig({
            'grid_zoom' : 0,
        })
        session = fo.launch_app(config=config)

        print('Processing matches ...')
        msg = 'Matching instructions:\n'
        msg += '\t0 = Merge\n'
        msg += '\t1 = Noisy 1\n'
        msg += '\t2 = Noisy 2\n'
        msg += '\t12 = Noisy 1 and 2\n'
        msg += '\tEnter = Ignore\n'
        msg += '\t3 = Skip cluster\n'
        msg += '\t-1 = Exit'
        print(msg)

        try:
            labels_map = {}
            ignore_pairs = []

            skip = self.load_from_csv(self.skip_path)
            noisy = self.load_from_csv(self.noisy_path)

            for i, (labels, dst) in enumerate(matches.items()):

                x, y = labels.split(',')

                if x in skip:
                    print(f'Skipping {x}')
                    continue
                if x in noisy or y in noisy:
                    print(f'Skipping noisy {[x, y]}')
                    continue

                xview = dataset.filter_labels("ground_truth", F("label") == x).take(25)
                yview = dataset.filter_labels("ground_truth", F("label") == y).take(25)
                sample_ids = [
                    *[s.id for s in xview.select_fields('id')],
                    *[s.id for s in yview.select_fields('id')]
                ]
                selected_view = (
                    dataset
                    .select(sample_ids)
                    .sort_by(F("ground_truth.label"))
                )
                session.view = selected_view

                inp = input('{}. Merge {} @ {} : '.format(i, [x, y], dst))

                if inp == '0':
                    labels_map[y] = x
                    skip[y] = '1'
                    self.append_to_csv(self.skip_path, [y])
                    print(f'Merged {[x, y]}')

                elif inp == '1':
                    labels_map[x] = '-1'
                    noisy[x] = '1'
                    self.append_to_csv(self.noisy_path, [x])
                    print(f'Noisy {x}')

                elif inp == '2':
                    labels_map[y] = '-1'
                    noisy[y] = '1'
                    self.append_to_csv(self.noisy_path, [y])
                    print(f'Noisy {y}')

                elif inp == '12':
                    labels_map[x] = '-1'
                    labels_map[y] = '-1'
                    noisy[x] = '1'
                    noisy[y] = '1'
                    self.append_to_csv(self.noisy_path, [x, y])
                    print(f'Noisy {[x, y]}')

                elif inp == '3':
                    skip[x] = '1'
                    self.append_to_csv(self.skip_path, [x])
                    print(f'Skipping to next label')

                elif inp == '-1':
                    print(f'Exiting')
                    break

                else:
                    ignore_pairs.append(labels)
                    print('Nothing doing')

                if labels_map:
                    self.save_matches(labels_map)
                    labels_map = {}

                if ignore_pairs:
                    self.save_ignored(ignore_pairs)
                    ignore_pairs = []

        finally:

            ignore = sorted(set(ignore_pairs))

            while True:
                x = input('Matches processed. Confirm [Y/N] : ').lower()
                if x == 'y':
                    labels_map = self.load_matches()
                    ignore = self.load_ignored()
                    break
                elif x == 'n':
                    labels_map = {}
                    ignore = []
                    break
            
            if len(labels_map) or len(ignore):

                x = input('Finalize? [Y/N] : ').lower()
                if x == 'y':
                    final = True
                else:
                    final = False

                print('Updating annotations ...')
                new_annotations = self.merger.update_annotations(
                    annotations, labels_map, ignore, final)
                
                print(f'Saving annotations at {self.new_annotations_path}')
                jwrite(self.new_annotations_path, new_annotations)
            
                print('Deleting temp files ...')
                if self.matches_path.exists():
                    self.matches_path.unlink()
                if self.ignore_path.exists():
                    self.ignore_path.unlink()
                if self.skip_path.exists():
                    self.skip_path.unlink()
                if self.noisy_path.exists():
                    self.noisy_path.unlink()

            else:
                print('No updated labels found.')

            print('Waiting for app to exit ...')
            session.wait()

            print('Closing FiftyOne Annotation ...')
            dataset.delete()


    def load_dataset(self):
        
        print('Loading dataset ... ')
        print(f'Faces dir: {self.faces_path}')
        print(f'Labels dir: {self.annotations_path}')

        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.FiftyOneImageClassificationDataset,
            data_path=self.faces_path,
            labels_path=self.annotations_path,
            name=self.name,
        )
        print(dataset)

        return dataset


    def save_matches(self, matches):

        if not self.matches_path.exists():
            columns = ['Old Label', 'New Label']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.matches_path, index=False)

        if matches:
            data = [(k, v) for k, v in matches.items()]
            df = pd.DataFrame(data)
            df.to_csv(self.matches_path, mode='a', header=False, index=False)

    def load_matches(self):

        matches = []
        if self.matches_path.exists():
            df = pd.read_csv(self.matches_path)
            matches = df.values
        
        return matches


    def save_ignored(self, ignore):

        if not self.ignore_path.exists():
            columns = ['Key']
            df = pd.DataFrame(columns=columns)
            df.to_csv(self.ignore_path, index=False)

        if ignore:
            df = pd.DataFrame(ignore)
            df.to_csv(self.ignore_path, mode='a', header=False, index=False)

    def load_ignored(self):

        ignore = []
        if self.ignore_path.exists():
            df = pd.read_csv(self.ignore_path)
            ignore = df['Key'].to_list()
        
        return ignore


    def load_from_csv(self, csv_path):
        # Must be called before save_to_csv as it creates csv file
        data = {}
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            data = {str(k): '1' for k in df['Key'].to_list()}
        else:
            columns = ['Key']
            df = pd.DataFrame(columns=columns)
            df.to_csv(csv_path, index=False)

        return data

    def append_to_csv(self, csv_path, data):
        df = pd.DataFrame(data)
        df.to_csv(csv_path, mode='a', header=False, index=False)
