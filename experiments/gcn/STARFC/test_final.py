import os
import gc
import time
import torch
import linecache
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from tqdm import tqdm
from mmcv import Config
from scipy.sparse import csr_matrix
from evaluation.evaluate import evaluate
from src.models.gcn import HEAD, HEAD_test
from src.models import build_model
from src.datasets import build_dataset
from utils import Timer
from utils import sparse_mx_to_torch_sparse_tensor
from utils import build_knns
from utils import fast_knns2spmat
from utils import build_symmetric_adj
from utils import row_normalize
from utils import mkdir_if_no_exists
from utils import list2dict
from utils import write_meta
from utils.misc import l2norm


def _find_parent(parent, u):
    idx = []
    # parent is a fixed point
    while (u != parent[u]):
        idx.append(u)
        u = parent[u]
    for i in idx:
        parent[i] = u
    return u


def edge_to_connected_graph(edges, num):
    parent = list(range(num))
    for u, v in edges:
        p_u = _find_parent(parent, u)
        p_v = _find_parent(parent, v)
        parent[p_u] = p_v

    for i in range(num):
        parent[i] = _find_parent(parent, i)
    remap = {}
    uf = np.unique(np.array(parent))
    for i, f in enumerate(uf):
        remap[f] = i
    cluster_id = np.array([remap[f] for f in parent])
    return cluster_id



dir = 'E:/StreamFace/Data/data'
names = [
    'abcnews', 'abcnewsaus', 'africanews', 'aljazeera', 'arirang',
    'bloombergqt', 'cgtnnews', 'channelstv','cna', 'dwnews', 'euronews',
    'france24', 'gbnews', 'nbc2news', 'nbcnow', 'ndtv', 'news12', 'ptvworld',
    'rtnews', 'skynews', 'trtworld', 'wion'
]

for name in names:

	print("\n{} {} {}\n".format("="*20, name, "="*20), flush=True)

	cfg = Config.fromfile("./cfg_train_star.py")
	cfg.eval_interim = False

	prefix = f'{dir}/{name}/dataset/gcn'
	work_dir = f'{dir}/{name}/clustering/work_dir/starfc'

	target = "test"
	feature_path = os.path.join(prefix, 'features')
	knn_path = os.path.join(prefix, 'knns', target, 'faiss_k_80.npz')

	#model_path_list=['train_model_sample7']
	#backbone_index=['4299']

	for model_i in [0]:
		model_i = int(model_i)
		model_path = os.path.join(work_dir, 'models')
		print('model_path', model_path)
		"""
		# Select latest model
		models = sorted([os.path.join(model_path, f) for f in os.listdir(model_path)],
						key=os.path.getmtime, reverse=True)
		backbone_name = os.path.basename(models[0])
		HEAD_name = os.path.basename(models[1])
		"""
		backbone_name = "Backbone.pth"
		HEAD_name = "Head.pth"
		print(backbone_name)
		print(HEAD_name)
		use_cuda = True
		use_gcn = True

		if use_gcn:
			knns = np.load(knn_path, allow_pickle=True)['data']
			nbrs = knns[:, 0, :]
			dists = knns[:, 1, :]
			edges = []
			score = []
			inst_num = knns.shape[0]
			print("inst_num:", inst_num)

			feature_path = os.path.join(feature_path, target)

			# print(**cfg.model['kwargs'])
			model = build_model('gcn', **cfg.model['kwargs'])
			model.load_state_dict(torch.load(os.path.join(model_path, backbone_name)))
			HEAD_test1 = HEAD_test(nhid=512)
			HEAD_test1.load_state_dict(torch.load(os.path.join(model_path, HEAD_name)), False)

			with Timer('build dataset'):
				for k, v in cfg.model['kwargs'].items():
					setattr(cfg.test_data, k, v)
				cfg.test_data['feat_path'] = os.path.join(prefix, cfg.test_data['feat_path'])
				cfg.test_data['label_path'] = os.path.join(prefix, cfg.test_data['label_path'])
				cfg.test_data['knn_graph_path'] = os.path.join(prefix, cfg.test_data['knn_graph_path'])
				dataset = build_dataset(cfg.model['type'], cfg.test_data)

			features = torch.FloatTensor(dataset.features)
			adj = sparse_mx_to_torch_sparse_tensor(dataset.adj)
			if not dataset.ignore_label:
				labels = torch.FloatTensor(dataset.gt_labels)

			with Timer('NN pairs'):
				pair_a = []
				pair_b = []
				pair_a_new = []
				pair_b_new = []
				for i in range(inst_num):
					pair_a.extend([int(i)] * 80)
					pair_b.extend([int(j) for j in nbrs[i]])
				for i in range(len(pair_a)):
					if pair_a[i] != pair_b[i]:
						pair_a_new.extend([pair_a[i]])
						pair_b_new.extend([pair_b[i]])
				pair_a = pair_a_new
				pair_b = pair_b_new
				pair_num = len(pair_a)

			print("num_pairs: ", pair_num)

			if use_cuda:
				model.cuda()
				HEAD_test1.cuda()
				features = features.cuda()
				adj = adj.cuda()
				labels = labels.cuda()

			model.eval()
			HEAD_test1.eval()
			test_data = [[features, adj, labels]]

			with Timer('First-0 step'):
				with torch.no_grad():

					output_feature = model(test_data[0])

					score = []

					patch_size = 500000
					patch_num = int(np.ceil(pair_num / patch_size))
					#print("patch_size:", patch_size)
					#print("patch_num:", patch_num)
					for i in tqdm(range(patch_num)):
						id1 = pair_a[i * patch_size:(i + 1) * patch_size]
						id2 = pair_b[i * patch_size:(i + 1) * patch_size]

						score_ = HEAD_test1(output_feature[id1],output_feature[id2])

						score.extend(score_)

			score = np.array(score)
			pair_a = np.array(pair_a)
			pair_b = np.array(pair_b)

			for threshold1 in [0.5, 0.6, 0.7, 0.8]:
				with Timer('Inference'):
					with Timer('First step'):

						idx = np.where(score > threshold1)[0].tolist()
						id1 = np.array([pair_a[idx].tolist()])
						id2 = np.array([pair_b[idx].tolist()])
						edges = np.concatenate([id1, id2], 0).transpose()
						value = [1] * len(edges)

						adj2 = csr_matrix((value, (edges[:,0].tolist(), edges[:,1].tolist())), shape=(inst_num, inst_num))
						link_num = np.array(adj2.sum(axis=1))
						common_link = adj2.dot(adj2)

					for threshold2 in [0.52, 0.62, 0.72, 0.82]:
						with Timer('Second step'):
							edges_new = []
							edges = np.array(edges)
							share_num = common_link[edges[:,0].tolist(), edges[:,1].tolist()].tolist()[0]
							edges = edges.tolist()

							for i in range(len(edges)):
								if ((link_num[edges[i][0]]) != 0) & ((link_num[edges[i][1]]) != 0):
									if max((share_num[i])/link_num[edges[i][0]],(share_num[i])/link_num[edges[i][1]])>threshold2:
										edges_new.append(edges[i])
								if i%10000000==0:
									print(i)

						with Timer('Last step'):
							pred_labels = edge_to_connected_graph(edges_new, inst_num)

						# save clustering results
						idx2lb = list2dict(pred_labels, ignore_value=-1)
						write_labels_path = os.path.join(work_dir, f'pred_labels_{threshold1}_{threshold2}.txt')
						write_meta(write_labels_path, idx2lb)

						gt_labels = labels.detach().cpu().numpy().astype(np.int32)

						print('the threshold1 is:{}'.format(threshold1))
						print('the threshold2 is:{}'.format(threshold2))
						print('# clusters: {}'.format(len(np.unique(pred_labels))))
						evaluate(gt_labels, pred_labels, 'pairwise')
						evaluate(gt_labels, pred_labels, 'bcubed')
						evaluate(gt_labels, pred_labels, 'nmi')
