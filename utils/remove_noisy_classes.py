from pathlib import Path
from tqdm import tqdm
from shutil import move
from streamface.streamface.utils import jread, jwrite


dir = Path('./data')
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]

for name in names:
    print(name)
    noisy_path = dir / name / "noisy_faces"
    noisy_path.mkdir(exist_ok=True)
    annotation_path = dir / name / "metadata" / "annotations_manual_new.json"
    annotations = jread(annotation_path)['labels']
    new_annotations = {}
    print("Before: ", len(annotations))
    for k, v in tqdm(annotations.items()):
        if v['label'] >= 0:
            new_annotations[k] = v
        else:
            try:
                src = dir / name / "faces" / f"{k}.jpg"
                move(str(src), str(noisy_path))
            except Exception as e:
                print(e)
    print("After: ", len(new_annotations))
    new_annotation_path = dir / name / "metadata" / "annotations_manual_new.json"
    new_annotations = {'labels': new_annotations}
    jwrite(new_annotation_path, new_annotations)
