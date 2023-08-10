import pandas as pd
from pathlib import Path


def get_image_files(
        dir: str,
    ) -> list:

    path = Path(dir)
    filenames = sorted([
        fp.stem for fp in path.iterdir() if fp.suffix == '.png'
    ])

    return filenames


if __name__ == "__main__":

    dir = ''
    names = [
        'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
        'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
        'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
        'rtnews', 'skynews', 'trtworld', 'wion'
    ]

    for i, name in enumerate(names):
        print('Processing {}: '.format(name), end='', flush=True)
        try:
            path = Path(dir) / name / 'frames'
            filenames = get_image_files(path)
            timestamps = [x.split('_')[2] for x in filenames]
            df = pd.DataFrame(timestamps, columns=['Timestamp'])
            df.to_csv('./{}_frames.csv'.format(name), index=None)
            print('SUCCESS')
        except:
            print('FAILED')
    