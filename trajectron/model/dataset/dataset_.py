import os
from copy import deepcopy

import dill
import numpy as np
import numpy.random as nr
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
        self.kalman_classes = list()
        self.class_count_dict = list()

        self._augment = False
        self.scores_path = scores_path
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            node_type_dataset = NodeTypeDatasetKalman(env, scores_path, node_type, state, pred_state, node_freq_mult,
                                                      scene_freq_mult, hyperparams, **kwargs)
            self.node_type_datasets.append(node_type_dataset)
            self.kalman_classes.append(node_type_dataset.kalman_classes)
            self.class_count_dict.append(node_type_dataset.class_count_dict)

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
        self.rebalance_bins()
        # self.smote()
        # import pdb; pdb.set_trace()

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

    # def smote(self):
    #     # data -> (first_history_index, x_t, y_t, x_st_t, y_st_t,
    #     # neighbors_data_st, neighbors_edge_value, robot_traj_st_t, map_tuple)
    #     # TODO: find a way to use SMOTE -> reason on the encoded tensor?
    #     n_class = len(self.class_count_dict)
    #     targets = self.kalman_classes
    #     n_max = max(self.class_count_dict.values())
    #     aug_data = []
    #     aug_label = []

    #     for k in range(1, n_class):
    #         indices = np.where(targets == k)[0]
    #         class_len = len(indices)
    #         class_dist = np.zeros((class_len, class_len))
    #         import pdb; pdb.set_trace()
    #         class_data = [self.__getitem__(i) for i in indices]
    #         # Augmentation with SMOTE ( k-nearest )
    #         for i in range(class_len):
    #             for j in range(class_len):
    #                 class_dist[i, j] = np.linalg.norm(class_data[i] - class_data[j])
    #         sorted_idx = np.argsort(class_dist)

    #         for i in range(n_max - class_len):
    #             lam = nr.uniform(0, 1)
    #             row_idx = i % class_len
    #             col_idx = int((i - row_idx) / class_len) % (class_len - 1)
    #             new_data = np.round(lam * class_data[row_idx] + (1 - lam) * class_data[sorted_idx[row_idx, 1 + col_idx]])
    #             aug_data.append(new_data.astype('uint8'))
    #             aug_label.append(k)
    #     return np.array(aug_data), np.array(aug_label)

    def rebalance_bins(self):
        # TODO Use 1 spaced clusters
        env_name = self.env.scenes[0].name
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
            dic_ = deepcopy(dic)
            sum_ = 0
            done = False
            i = lbls.max()
            while i > 0 and not done:  # left 0.7 percent
                if sum_ + dic_[i] >= scores.shape[0] * 0.007:
                    done = True
                else:
                    sum_ += dic_[i]
                    del (dic_[i])
                    i -= 1
            dic_[i + 1] = sum_

            original_keys = dic_.keys()
            original_keys = list(original_keys)
            new_keys = sorted(original_keys, key=lambda x: dic_[x], reverse=True)
            switched_dic = {new_keys[k]: k for k in range(len(original_keys))}
            minority_class = i + 1
            assert sum(dic_.values()) == scores.shape[0]
            class_count = [*dic_.values()]
            class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
            for l in range(len(lbls)):
                if lbls[l] > minority_class:
                    lbls[l] = minority_class
            for l in range(len(lbls)):
                lbls[l] = switched_dic[lbls[l]]

            dic_compare = {}
            for i in range(lbls.max() + 1):
                dic_compare[i] = 0
            for l in lbls:
                dic_compare[l] += 1

            self.class_weights_all = class_weights[lbls]
            balanced_class_weights = (1 - beta) / (1 - torch.pow(beta, torch.tensor(class_count, dtype=torch.float)))
            self.balanced_class_weights_all = balanced_class_weights[lbls]
            self.weighted_sampler = data.WeightedRandomSampler(
                weights=self.class_weights_all,
                num_samples=len(self.class_weights_all),
                replacement=True
            )

            self.kalman_classes = lbls
            self.class_count_dict = dic_compare

    def load_scores(self):
        env_name = self.env.scenes[0].name
        import dill
        with open(os.path.join(self.scores_path, '%s_kalman.pkl' % env_name), 'rb') as f:
            self.scores = dill.load(f)
        # with open('/home/makansio/raid21/Trajectron-EWTA/experiments/pedestrians/%s_deter_multi.pkl' % env_name, 'rb') as f:
        #     self.scores = dill.load(f)
        assert self.scores.shape[0] == len(self.index), 'Loaded scores should match the current dataset (%d vs %d)' % (
            self.scores.shape[0], len(self.index))

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        sample = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                        self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.kalman_classes[i],)
        # scene = scene.augment()
        # node = scene.get_node_by_id(node.id)
        # sample_aug = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
        #                                 self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.scores[i],)
        return sample
