import json


def labels2annotations(filenames_path, labels_path, output_path):

    filenames = _read_items(filenames_path)
    labels = _read_items(labels_path)
    assert len(filenames) == len(labels)

    annotations = {
        'labels' : {
            name : {'label' : label}
            for name, label in zip(filenames, labels)
        }
    }

    _jwrite(output_path, annotations)


def _read_items(filepath):
    with open(filepath, 'r') as fp:
        items = fp.read().splitlines()
    return items


def _jwrite(filepath, data):
    with open(filepath, 'w', encoding='utf-8') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)


if __name__ == '__main__':

    filenames_path = '../../data/sky/gcn/filenames/train.txt'
    labels_path = '../../data/sky/gcn/labels/train.meta'
    output_path = '../../data/sky/gcn/annotations.json'

    labels2annotations(
        filenames_path=filenames_path,
        labels_path=labels_path,
        output_path=output_path,
    )