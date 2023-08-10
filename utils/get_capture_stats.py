import pandas as pd

from pathlib import Path
from datetime import datetime
from collections import Counter


def get_image_files(dir):
    def is_imgfile(fp):
        ret = True if fp.is_file() and fp.suffix in ['.png', '.jpg'] else False
        return ret

    path = Path(dir)
    filenames = sorted([
        fp.stem for fp in path.iterdir() if is_imgfile(fp)
    ])

    return filenames


def create_dataframe(filenames, output_path):
    datapoints = []
    for filename in filenames:
        channel, _, timestamp = filename.split('_')
        datapoint = channel + '_' + timestamp[:11] + '0'
        datapoints.append(datapoint)

    frequency_dict = Counter(datapoints)
    
    data = []
    for name, count in frequency_dict.items():
        channel, timestamp = name.split('_')
        ts_datetime = datetime.strptime(timestamp, '%Y%m%d%H%M')
        date = ts_datetime.date()
        time = ts_datetime.time()
        timestamp = datetime.strftime(ts_datetime, '%Y-%m-%d %H:%M')
        data.append((channel, date, time, timestamp, count))
    
    df = pd.DataFrame(data, columns=['Channel', 'Date', 'Time', 'Timestamps', '# Frames'])
    df.to_csv(output_path)


input_dir = './data/dwnews/frames'
output_path = './stats_capture/dwnews_capture_stats.csv'

print(f'Reading image files from {input_dir} ... ', end='', flush=True)

filenames = get_image_files(input_dir)

print('DONE', flush=True)

print(f'Creating DataFrame at {output_path} ... ', end='', flush=True)

create_dataframe(filenames, output_path)

print('DONE', flush=True)
