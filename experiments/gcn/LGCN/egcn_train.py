import os

cfg_name='cfg_train_gcne'
config='./cfg_train_gcne.py'
load_from=f'./data/work_dir/{cfg_name}/latest.pth'


# train
cmd1 = f'python vegcn/main.py \
    --config {config} \
    --phase train'

# test
cmd2 = f'python vegcn/main.py \
    --config {config} \
    --phase test \
    --load_from {load_from} \
    --save_output \
    --force'


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['PYTHONPATH'] = '.'

os.system(cmd1)

#os.system(cmd2)
