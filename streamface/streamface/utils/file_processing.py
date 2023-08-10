import json
import pickle

from pathlib import Path
from tqdm import tqdm


def jwrite(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)

def jread(filepath):
    with open(filepath, 'r', encoding='utf-8') as fp:
        obj = json.load(fp)
    return obj


def pkread(filepath):
    with open(filepath, 'rb') as fp:
        obj = pickle.load(fp)
    return obj

def pkwrite(filepath, obj):
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)

def pkmerge(dirpath):
    def is_pklfile(fp):
        ret = True if fp.is_file() and fp.suffix in ['.pkl'] else False
        return ret

    filepaths = sorted(
        [fp for fp in Path(dirpath).iterdir() if is_pklfile(fp)],
        key=lambda x: x.stem
    )

    obj_list = []
    for filepath in tqdm(filepaths):
        obj = pkread(filepath)
        obj_list.append(obj)
    
    if isinstance(obj_list[0], dict):
        merged_objs = {k:v for obj in obj_list for k,v in obj.items()}
    else:
        try:
            for x in obj_list[0]:
                break
            merged_objs = [x for obj in obj_list for x in obj]
        except:
            print('Error: Each pickle file must contain a dict or an iterable.')
            return None
    
    return merged_objs
