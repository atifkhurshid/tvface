import os


config='./cfg_train_gcnv.py'

dir = 'E:/StreamFace/Data/data'
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]

for name in names:

    print("\n{} {} {}\n".format("="*20, name, "="*20), flush=True)

    prefix = f'{dir}/{name}/dataset/gcn'
    work_dir=f'{dir}/{name}/clustering/work_dir/vgcn'
    load_from=f'{work_dir}/latest.pth'

    # train
    cmd1 = f'python vegcn/main.py \
        --config {config} \
        --phase train \
        --prefix {prefix} \
        --work_dir {work_dir}'

    # test
    cmd2 = f'python vegcn/main.py \
        --config {config} \
        --phase test \
        --prefix {prefix} \
        --work_dir {work_dir} \
        --load_from {load_from} \
        --save_output \
        --force'


    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    os.environ['PYTHONPATH'] = '.'

    os.system(cmd1)

    os.system(cmd2)
