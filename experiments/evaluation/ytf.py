"""
Source: https://github.com/grigorisg9gr/polynomial_nets/blob/master/face_recognition/eval/lfw.py
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from pathlib import Path
from sklearn.preprocessing import normalize

from metrics import evaluate


class YTF(object):

    def __init__(self, dir, n_samples=32, marginx=0.3, marginy=0.2):

        self.dir = Path(dir)
        self.n_samples = n_samples
        self.marginx = marginx
        self.marginy = marginy
        self.pairs, self.issame_list = self._read_pairs(self.dir / 'splits.csv')
        self.paths = self._get_paths(self.dir, self.pairs)

        print('ytf', len(self.paths))
        print('same', self.issame_list.shape)


    def __getitem__(self, index):

        n_samples = min(self.n_samples, len(self.paths[index]))
        paths = random.sample(self.paths[index], n_samples)

        imgs = self._load_paths(paths)

        issame = self.issame_list[index // 2]

        return imgs, issame


    def __len__(self):

        return len(self.paths)


    def _read_pairs(self, pairs_filename):

        df = pd.read_csv(pairs_filename)
        df.columns = df.columns.str.strip()
        df['first name'] = df['first name'].str.strip()
        df['second name'] = df['second name'].str.strip()

        pairs = list(df['first name'].str.split('/') + df['second name'].str.split('/'))
        issame_list = np.array(list(df['is same'].astype(bool)))

        return pairs, issame_list


    def _get_paths(self, dir, pairs):

        nrof_skipped_pairs = 0
        path_list = []
        for pair in pairs:
            path0 = dir / pair[0] / pair[1]
            path1 = dir / pair[2] / pair[3]
            if path0.exists() and path1.exists():    # Only add the pair if both paths exist
                paths0 = list(path0.iterdir())
                paths1 = list(path1.iterdir())
                if len(paths0) and len(paths1):
                    path_list += (paths0, paths1)
                else:
                    print('file not exists', path0, path1)
                    nrof_skipped_pairs += 1         
            else:
                print('not exists', path0, path1)
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
        
        return path_list


    def _load_paths(self, paths):
        imgs = []
        for path in paths:
            img = np.asarray(Image.open(path))
            marginx = int(img.shape[1] * self.marginx)
            marginy = int(img.shape[0] * self.marginy)
            img = img[marginy:-marginy, marginx:-marginx, :]
            imgs.append(img)
        imgs = np.array(imgs)

        return imgs


def calculate_embeddings(model, faces, batch_size):

    embeddings = []
    for i in range(int(np.ceil(len(faces) / batch_size))):
        batch_data = faces[i*batch_size:(i+1)*batch_size]
        embs = model(batch_data)
        embeddings.extend(embs)
    embeddings = np.array(embeddings)

    # normalize(embeddings, norm='l2', axis=1, copy=False, return_norm=False)

    return embeddings


def test(dataset, model, batch_size):

    issame_list = dataset.issame_list

    embeddings = []
    for i in tqdm(range(len(dataset))):
        data, _ = dataset[i]
        embs = calculate_embeddings(model, data, batch_size)
        embedding = np.mean(embs, axis=0, keepdims=True)
        # normalize(embedding, norm='l2', axis=1, copy=False, return_norm=False)
        embeddings.extend(embedding)
    embeddings = np.array(embeddings)

    # embeddings = np.random.uniform(-1,1,(len(dataset), 512))

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

    data_dir = "D:/StreamFace/YTF"
    output_path = 'D:/StreamFace/YTF_results.xlsx'

    dataset = YTF(data_dir, 32, 0.3, 0.25)
    model = FaceRepresentation('arcface')

    names = [
        'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
        'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
        'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
        'rtnews', 'skynews', 'trtworld', 'wion'
    ]
    names = ['']

    with pd.ExcelWriter(output_path) as writer:

        startrow = 1

        for name in names:

            print(name)

            weights_path = f'D:/StreamFace/data/{name}/rawmodel/arcface_weights.h5'
            # weights_path = f'C:/Users/Vision2/.deepface/weights/arcface_weights.h5'
            try:
                model.model.model.load_weights(weights_path)
            except:
                print('Couldnt load weights from', weights_path)
                continue

            model_fn = model.getembeddings
            # model = None
            
            batch_size = 32

            tpr, fpr, acc, acc_std, tar, tar_std, far = test(dataset, model_fn, batch_size)

            plt.plot(fpr, tpr)
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

            pd.Series(name).to_excel(writer, sheet_name='YTF',
                startrow=startrow, startcol=1, index=False, header=False)
            startrow += 1

            df1.to_excel(writer, sheet_name='YTF', index=True, header=False, 
                startrow=startrow, startcol=1)
            startrow += len(df1)

            df2.to_excel(writer, sheet_name='YTF', index=True, header=False, 
                startrow=startrow, startcol=1)
            startrow += len(df2) + 3
        
    # plt.show()

    print()