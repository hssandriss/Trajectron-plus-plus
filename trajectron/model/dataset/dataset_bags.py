import os
from copy import deepcopy

import dill
import numpy as np
import torch
from torch.utils import data

from .preprocessing import get_node_timestep_data


class EnvironmentDatasetKalmanGroupExperts(object):
    def __init__(self, env, scores_path, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, bin_borders=None, nb_bins=4, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self.kalman_classes = list()
        self.scores = list()
        self.kalman_classes_bins = list()
        self.class_count_dict = list()
        self.class_weights = list()
        self.inv_class_weights = list()
        self.borders = list()
        self.bin_borders = bin_borders
        self.nb_bins = nb_bins
        self._augment = False
        self.scores_path = scores_path
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            node_type_dataset = NodeTypeDatasetKalmanGroupExperts(env, scores_path, node_type, state, pred_state, node_freq_mult,
                                                                  scene_freq_mult, self.bin_borders, self.nb_bins, hyperparams, **kwargs)
            self.node_type_datasets.append(node_type_dataset)
            self.kalman_classes.append(node_type_dataset.kalman_classes)
            self.kalman_classes_bins.append(node_type_dataset.kalman_classes_bins)
            self.class_count_dict.append(node_type_dataset.class_count_dict)
            self.borders.append(node_type_dataset.bin_borders)
            self.class_weights.append(node_type_dataset.balanced_class_weights)
            self.inv_class_weights.append(node_type_dataset.class_weights)
            self.scores.append(node_type_dataset.scores)

        self.train_borders_match_class_per_bin = None  # the classes for each group; 0 is always reserved for others
        if self.borders != None:
            self.train_borders_match_class_per_bin = {}
            for i, (k, v) in enumerate(self.borders[0].items()):
                self.train_borders_match_class_per_bin[k] = [i + 1 for i in range(len(v))]

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


class NodeTypeDatasetKalmanGroupExperts(data.Dataset):
    def __init__(self, env, scores_path, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, bin_borders, n_bins, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.n_bins = n_bins

        self.scores_path = scores_path
        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]
        self.load_scores()
        # self.rebalance_bins()
        self.bin_borders = bin_borders
        self.rebalance_bins()
        #import pdb; pdb.set_trace()

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

    def rebalance_bins(self):
        # Borders : tuple(int, int) boarders
        # TODO Use 1 spaced clusters
        env_name = self.env.scenes[0].name
        with open(os.path.join(self.scores_path, '%s_kalman.pkl' % env_name), 'rb') as f:
            scores = dill.load(f)
        self.scores = scores
        lbls_old = (scores / 0.5).astype(np.int)
        # Calculating class values counts
        dic_old = {}
        for i in range(lbls_old.max() + 1):
            dic_old[i] = 0
        for l in lbls_old:
            dic_old[l] += 1
        # delete and shift the classes that contain zero examples
        dic = {}
        dic_idx = 0
        match_dicold_dic = {}
        for k in range(lbls_old.max() + 1):
            if dic_old[k] != 0:
                dic[dic_idx] = dic_old[k]
                match_dicold_dic[k] = dic_idx
                dic_idx += 1
        lbls = np.zeros(len(lbls_old)).astype(int)
        for i in range(len(lbls_old)):
            lbls[i] = match_dicold_dic[lbls_old[i]]

        dic_bins = {}
        for i in range(self.n_bins + 1):
            dic_bins[i] = []
        k = 0
        while k <= lbls.max():
            if k == 0:
                if 0 in dic.keys() and 1 in dic.keys():
                    if dic[0] >= 2 * dic[1]:
                        dic_bins[0] = [0]
                    else:
                        if dic[0] > 5 * 10**self.n_bins:
                            while dic[k] >= 5 * 10**self.n_bins:
                                dic_bins[0].append(k)
                                k += 1
                        elif dic[0] > 5 * 10**(self.n_bins - 1):
                            while dic[k] >= 5 * 10**(self.n_bins - 1):
                                dic_bins[0].append(k)
                                k += 1

            for i in range(self.n_bins):
                if dic[k] >= 10**i and dic[k] < 10**(i + 1):
                    dic_bins[self.n_bins - i].append(k)
                    break
            k += 1

        if self.bin_borders == None:
            dic_bins = {}
            for i in range(self.n_bins + 1):
                dic_bins[i] = []
            k = 0
            while k <= lbls.max():
                if k == 0:
                    if 0 in dic.keys() and 1 in dic.keys():
                        if dic[0] >= 2 * dic[1]:
                            dic_bins[0] = [0]
                        else:
                            if dic[0] > 5 * 10**self.n_bins:
                                while dic[k] >= 5 * 10**self.n_bins:
                                    dic_bins[0].append(k)
                                    k += 1
                            elif dic[0] > 5 * 10**(self.n_bins - 1):
                                while dic[k] >= 5 * 10**(self.n_bins - 1):
                                    dic_bins[0].append(k)
                                    k += 1

                for i in range(self.n_bins):
                    if dic[k] >= 10**i and dic[k] < 10**(i + 1):
                        dic_bins[self.n_bins - i].append(k)
                        break
                k += 1

            lbls_bins = np.zeros(len(lbls)).astype(int)
            if env_name == 'univ_train' and self.n_bins == 4:
                dic_bins = {0: [0, 1],
                            1: [2, 3],
                            2: [4, 5, 6, 7, 8, 9],
                            3: [10, 11, 12, 13, 15, 16],
                            4: [14, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]}
            for c in range(self.n_bins + 1):
                #lbls_bins [np.where((lbls <= dic_bins[c][-1]) & (lbls >= dic_bins[c][0]))[0]] = c
                for l in range(len(lbls)):
                    if lbls[l] in dic_bins[c]:
                        lbls_bins[l] = c

            self.bin_borders = dic_bins

        else:
            # self.bin_borders are given
            if list(dic_old.keys())[-1] > self.bin_borders[self.n_bins][-1]:
                print('PROBLEM IN DATASET_ !!!!!!!!!!!!!!!!!!!!!!!!!')
                # TODO get these index and change the labels to be the same as self.bin_borders[self.n_bins][-1]
            else:
                lbls = lbls_old
                dic = dic_old

            lbls_bins = np.zeros(len(lbls)).astype(int)
            for c in range(self.n_bins + 1):
                #lbls_bins [np.where((lbls <= self.bin_borders[c][-1]) & (lbls >= self.bin_borders[c][0]))[0]] = c
                for l in range(len(lbls)):
                    if lbls[l] in self.bin_borders[c]:
                        lbls_bins[l] = c
        # Calculating class values counts after sorting
        dic_ = {}
        for i in range(lbls.max() + 1):
            dic_[i] = 0
        for l in lbls:
            dic_[l] += 1
        assert sum(dic_.values()) == scores.shape[0]

        # class weights .i.e sampling probability
        class_count = [*dic_.values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        self.class_weights_all = class_weights[lbls]
        self.weighted_sampler = data.WeightedRandomSampler(
            weights=self.class_weights_all,
            num_samples=len(self.class_weights_all),
            replacement=True)

        # class weights based on effective number of samples
        # https://arxiv.org/pdf/1901.05555.pdf
        n = scores.shape[0]
        beta = (n - 1) / n
        # import pdb; pdb.set_trace()
        self.balanced_class_weights = (1 - beta) / (1 - torch.pow(beta, torch.tensor(class_count, dtype=torch.float)))
        self.balanced_class_weights_all = self.balanced_class_weights[lbls]

        self.class_weights = class_weights
        self.kalman_classes = lbls
        self.kalman_classes_bins = lbls_bins
        self.class_count_dict = dic_

    def load_scores(self):
        env_name = self.env.scenes[0].name
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
                                        self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.kalman_classes_bins[i], self.kalman_classes[i], self.scores[i],)
        # scene = scene.augment()
        # node = scene.get_node_by_id(node.id)
        # sample_aug = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
        #                                 self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.scores[i],)
        return sample
