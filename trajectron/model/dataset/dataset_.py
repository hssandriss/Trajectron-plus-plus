import os

import dill
import numpy as np
import torch
from torch.utils import data

from .preprocessing import get_node_timestep_data


class EnvironmentDatasetKalman(object):
    def __init__(self, env, scores_path, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False
        self.scores_path = scores_path
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDatasetKalman(env, scores_path, node_type, state, pred_state, node_freq_mult,
                                                                 scene_freq_mult, hyperparams, **kwargs))

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDatasetKalman(data.Dataset):
    def __init__(self, env, scores_path, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.scores_path = scores_path
        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
        self.load_scores()
        self.rebalance()

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            counter = 0
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                        (scene.frequency_multiplier if scene_freq_mult else 1) *\
                        (node.frequency_multiplier if node_freq_mult else 1)
                    counter += 1
            # print(counter, scene.timesteps, scene_freq_mult, node_freq_mult)

        return index

    def rebalance(self):
        env_name = self.env.scenes[0].name
        import dill
        with open(os.path.join(self.scores_path, '%s_kalman.pkl' % env_name), 'rb') as f:
            scores = dill.load(f)
            n = scores.shape[0]
            beta = (n - 1) / n
            lbls = (scores / 0.5).astype(np.int)
            dic = {}
            for i in range(lbls.max() + 1):
                dic[i] = 0
            for l in lbls:
                dic[l] += 1
            class_count = [*dic.values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            self.class_weights_all = class_weights[lbls]
            balanced_class_weights = (1 - beta) / (1 - torch.pow(beta, torch.tensor(class_count, dtype=torch.float)))
            self.balanced_class_weights_all = balanced_class_weights[lbls]
            self.weighted_sampler = data.WeightedRandomSampler(
                weights=self.class_weights_all,
                num_samples=len(self.class_weights_all),
                replacement=True
            )

    def load_scores(self):
        env_name = self.env.scenes[0].name
        import dill
        with open(os.path.join(self.scores_path, '%s_kalman.pkl' % env_name), 'rb') as f:
            self.scores = dill.load(f)
        # with open('/home/makansio/raid21/Trajectron-EWTA/experiments/pedestrians/%s_deter_multi.pkl' % env_name, 'rb') as f:
        #     self.scores = dill.load(f)
        assert self.scores.shape[0] == len(self.index), 'Loaded scores should match the current dataset (%d vs %d)' % (self.scores.shape[0], len(self.index))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        sample = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                        self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.scores[i], self.balanced_class_weights_all[i],)
        # scene = scene.augment()
        # node = scene.get_node_by_id(node.id)
        # sample_aug = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
        #                                 self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.scores[i],)
        return sample
