import os

config='./cfg_train_star.py'

dir = 'E:/StreamFace/Data/data'
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]
names = ['abcnewsaus']

for name in names:

    print("\n{} {} {}\n".format("="*20, name, "="*20), flush=True)

    prefix = f'{dir}/{name}/dataset/gcn'
    work_dir=f'{dir}/{name}/clustering/work_dir/starfc'

    # train
    cmd = f'python src/main.py \
        --config {config} \
        --phase train \
        --prefix {prefix} \
        --work_dir {work_dir}'


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONPATH'] = '.'

    os.system(cmd)
