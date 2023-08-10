"""
Source: https://github.com/grigorisg9gr/polynomial_nets/blob/master/face_recognition/eval/lfw.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import normalize

from metrics import evaluate


class LFW(object):

    def __init__(self):
        pass

    def __new__(cls, dir, margin=64):

        pairs = cls._read_pairs(os.path.join(dir, 'pairs.txt'))
        paths, issame_list = cls._get_paths(dir, pairs, 'jpg')
        data = []
        for path in tqdm(paths):
            img = np.asarray(Image.open(path))[margin:-margin, margin:-margin, :]
            data.append(img)
        data = np.array(data)
        issame_list = np.array(issame_list)
        print('lfw', data.shape)
        print('same', issame_list.shape)

        return (data, issame_list)


    def _read_pairs(pairs_filename):

        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        return pairs


    def _get_paths(dir, pairs, file_ext):

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = os.path.join(dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path1 = os.path.join(dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
                path_list += (path0, path1)
                issame_list.append(issame)
            else:
                print('not exists', path0, path1)
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs>0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
        
        return path_list, issame_list


def calculate_embeddings(model, faces, batch_size):

    embeddings = []
    for i in tqdm(range(int(np.ceil(len(faces) / batch_size)))):
        batch_data = faces[i*batch_size:(i+1)*batch_size]
        embs = model(batch_data)
        embeddings.extend(embs)
    embeddings = np.array(embeddings)

    normalize(embeddings, norm='l2', axis=1, copy=False, return_norm=False)

    return embeddings


def test(dataset, model, batch_size):

    data = dataset[0]
    issame_list = dataset[1]

    embeddings = calculate_embeddings(model, data, batch_size)
    # embeddings = np.random.uniform(-1,1,(data.shape[0], 512))

    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    tpr, fpr, accuracy, tar, tar_std, far = evaluate(
        embeddings1, embeddings2, issame_list, nrof_folds=10, far_target=1e-3)

    acc, acc_std = np.mean(accuracy), np.std(accuracy)

    return tpr, fpr, acc, acc_std, tar, tar_std, far


if __name__ == "__main__":

    import os
    import sys
    sys.path.append(os.getcwd())

    from streamface.streamface.face_representation import FaceRepresentation

    data_dir = "D:/StreamFace/LFW"
    output_path = 'D:/StreamFace/LFW_results.xlsx'

    dataset = LFW(data_dir, margin=69)
    model = FaceRepresentation('arcface')

    names = [
        'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
        'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
        'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
        'rtnews', 'skynews', 'trtworld', 'wion'
    ]
    names = ['abcnews', 'abcnewsaus', 'africanews', 'arirang', 'wion']
    names = ['']
    with pd.ExcelWriter(output_path) as writer:

        startrow = 1

        for name in names:

            print(name)

            weights_path = f'D:/StreamFace/data/{name}/rawmodel/arcface_weights.h5'

            try:
                model.model.model.load_weights(weights_path)
            except:
                print('Couldnt load weights from', weights_path)
                continue

            model_fn = model.getembeddings
            # model = None
            
            batch_size = 64

            tpr, fpr, acc, acc_std, tar, tar_std, far = test(dataset, model_fn, batch_size)

            x = fpr.copy()
            x[x == 0] = 1e-6
            y = tpr.copy()
            plt.plot(x, y)
            plt.xscale('log')
            plt.show(block=False)
            print('TPR: %2.5f, FPR: %2.5f' % (np.mean(tpr), np.mean(fpr)))
            print('Accuracy: %2.5f+-%2.5f' % (acc, acc_std))
            print('TAR: %2.5f+-%2.5f @ FAR=%2.5f' % (tar, tar_std, far))

            df1 = pd.DataFrame.from_dict({
                'TPR': tpr,
                'FPR': fpr,
            }, orient='index')

            df2 = pd.DataFrame.from_dict({
                'ROC Accuracy (Mean)': [acc],
                'ROC Accuracy (Std)': [acc_std],
                'TAR (Mean)': [tar],
                'TAR (Std)': [tar_std],
                'FAR': [far],
            }, orient='index')

            pd.Series(name).to_excel(writer, sheet_name='LFW',
                startrow=startrow, startcol=1, index=False, header=False)
            startrow += 1

            df1.to_excel(writer, sheet_name='LFW', index=True, header=False, 
                startrow=startrow, startcol=1)
            startrow += len(df1)

            df2.to_excel(writer, sheet_name='LFW', index=True, header=False, 
                startrow=startrow, startcol=1)
            startrow += len(df2) + 3

    print()