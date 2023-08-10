import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path

from annotations_processing import read_frames
from annotations_processing import read_annotations
from annotations_processing import create_annotations_dataframe
from timestamp_processing import collection_over_time
from class_processing import class_membership_distribution
from pose_processing import pose_angle_distribution


if __name__ == "__main__":

    input_dir = './data'
    frames_input_dir = "./data"
    output_path = './Dataset Statistics.xlsx'
    names = [
        'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
        'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
        'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
        'rtnews', 'skynews', 'trtworld', 'wion']

    print('Processing annotations ...')

    frames_dfs = {}
    annotations_dfs = {}

    for name in tqdm(names):
        try:
            frames_path = Path(frames_input_dir) / name / 'metadata' / f'{name}_frames.csv'
            frames_dfs[name] = read_frames(frames_path)

            annotations_path = Path(input_dir) / name / 'metadata' / 'annotations_manual_new.json'
            annotations_dict = read_annotations(annotations_path)
            annotations_dfs[name] = create_annotations_dataframe(annotations_dict)

        except Exception as e:
            print(e)
    
    # Combine all dataframes
    frames_dfs['total'] = pd.concat(list(frames_dfs.values()), axis=0, ignore_index=True)
    annotations_dfs['total'] = pd.concat(list(annotations_dfs.values()), axis=0, ignore_index=True)

    print('Frames and face detections over time ...')

    frames_over_time = {}
    detections_over_time = {}
    num_frames = {}
    num_detections = {}
    num_clusters = {}

    for name in tqdm(annotations_dfs):

        frames_timestamps = frames_dfs[name]['Timestamp']
        frames_over_time[name] = collection_over_time(frames_timestamps, unit='days')

        faces_timestamps = annotations_dfs[name]['Timestamp']
        detections_over_time[name] = collection_over_time(faces_timestamps, unit='days')

        num_frames[name] = np.sum(list(frames_over_time[name].values()))
        num_detections[name] = np.sum(list(detections_over_time[name].values()))
        num_clusters[name] = len(set(annotations_dfs[name]['Label']))

    # Align Frames and Detections for all Channels
    all_frames_timestamps = [x for name in annotations_dfs for x in frames_over_time[name]]
    all_detections_timestamps = [x for name in annotations_dfs for x in detections_over_time[name]]
    combined_timestamps = sorted(set(all_frames_timestamps + all_detections_timestamps))
    for name in tqdm(annotations_dfs):
        for ts in combined_timestamps:
            frames_over_time[name][ts] = frames_over_time[name].get(ts, 0)
            detections_over_time[name][ts] = detections_over_time[name].get(ts, 0)

    print('Processing manual annotations ...')

    manual_annotations_dfs = {}
    num_faces = {}
    num_classes = {}

    for name, df in tqdm(annotations_dfs.items()):

        manual_annotations_dfs[name] = df[df['Label'] >= 0]

        num_faces[name] = len(manual_annotations_dfs[name])
        num_classes[name] = len(set(manual_annotations_dfs[name]['Label']))

    print('Processing distributions ...')

    class_distributions = {}
    mask_distributions = {}
    gender_distributions = {}
    age_distributions = {}
    race_distributions = {}
    expression_distributions = {}
    yaw_distributions = {}
    pitch_distributions = {}
    roll_distributions = {}

    for name, df in tqdm(manual_annotations_dfs.items()):

        class_distributions[name] = class_membership_distribution(df['Label'])

        mask_distributions[name] = {
            k: v for k, v in sorted(df['Mask'].value_counts(sort=False).to_dict().items(),
            key=lambda x: x[0])}

        gender_distributions[name] = {
            k: v for k, v in sorted(df['Gender'].value_counts(sort=False).to_dict().items(),
            key=lambda x: x[0])} 

        age_distributions[name] = {
            k: v for k, v in sorted(df['Age'].value_counts(sort=False).to_dict().items(),
            key=lambda x: int(x[0].replace('+', '-').split('-')[0]))}

        race_distributions[name] = {
            k: v for k, v in sorted(df['Race'].value_counts(sort=False).to_dict().items(),
            key=lambda x: x[0])}

        expression_distributions[name] = {
            k: v for k, v in sorted(df['Expression'].value_counts(sort=False).to_dict().items(),
            key=lambda x: x[0])}

        yaw_distributions[name] = pose_angle_distribution(df['Yaw'])

        pitch_distributions[name] = pose_angle_distribution(df['Pitch'])

        roll_distributions[name] = pose_angle_distribution(df['Roll'])

    print('Writing statistics to {} ...'.format(output_path))

    with pd.ExcelWriter(output_path) as writer:

        overview_df = pd.DataFrame({
            'Names' : list(manual_annotations_dfs.keys()),
            'Frames' : list(num_frames.values()),
            'Detections': list(num_detections.values()),
            'Faces': list(num_faces.values()),
            'Clusters': list(num_clusters.values()),
            'Classes': list(num_classes.values()),
        })
        overview_df.to_excel(writer, index=False, sheet_name='Overview', startrow=1, startcol=1)

        for name in tqdm(manual_annotations_dfs):

            startrow = 1

            over_time_df = pd.DataFrame({
                'Timestamp': list(frames_over_time[name].keys()),
                'Frame Count': list(frames_over_time[name].values()),
                'Detection Count': list(detections_over_time[name].values()),
            })
            over_time_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(over_time_df) + 3

            class_distributions_df = pd.DataFrame({
                'Class Bins': list(class_distributions[name].keys()),
                'Bin Count': list(class_distributions[name].values())
            })
            class_distributions_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(class_distributions_df) + 3

            mask_distributions_df = pd.DataFrame({
                'Mask Cats': list(mask_distributions[name].keys()),
                'Mask Count': list(mask_distributions[name].values())
            })
            mask_distributions_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(mask_distributions_df) + 3

            gender_distributions_df = pd.DataFrame({
                'Gender Cats': list(gender_distributions[name].keys()),
                'Gender Count': list(gender_distributions[name].values())
            })
            gender_distributions_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(gender_distributions_df) + 3

            age_distributions_df = pd.DataFrame({
                'Age Cats': list(age_distributions[name].keys()),
                'Age Count': list(age_distributions[name].values())
            })
            age_distributions_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(age_distributions_df) + 3

            race_distributions_df = pd.DataFrame({
                'Race Cats': list(race_distributions[name].keys()),
                'Race Count': list(race_distributions[name].values())
            })
            race_distributions_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(race_distributions_df) + 3

            expression_distributions_df = pd.DataFrame({
                'Expression Cats': list(expression_distributions[name].keys()),
                'Expression Count': list(expression_distributions[name].values())
            })
            expression_distributions_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(expression_distributions_df) + 3

            pose_distributions_df = pd.DataFrame({
                'Angles': list(yaw_distributions[name].keys()),
                'Yaw': list(yaw_distributions[name].values()),
                'Pitch': list(pitch_distributions[name].values()),
                'Roll': list(roll_distributions[name].values()),
            })
            pose_distributions_df.to_excel(writer, index=False, sheet_name=name,
                startrow=startrow, startcol=1)
            startrow += len(pose_distributions_df) + 3


    print()