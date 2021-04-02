import os
from copy import deepcopy

import dill
import numpy as np
import numpy.random as nr
import torch
from torch.utils import data

from .preprocessing import get_node_timestep_data


class EnvironmentDatasetKalman(object):
    def __init__(self, env, scores_path, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, stack_right=0.007, borders=None, predifined_num_classes=None, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self.kalman_classes = list()
        self.class_count_dict = list()
        self.class_weights = list()
        self.inv_class_weights = list()
        self.boarders = list()
        self._augment = False
        self.scores_path = scores_path
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            node_type_dataset = NodeTypeDatasetKalman(env, scores_path, node_type, state, pred_state, node_freq_mult,
                                                      scene_freq_mult, hyperparams, stack_right=stack_right, predifined_num_classes=predifined_num_classes, **kwargs)
            self.node_type_datasets.append(node_type_dataset)
            self.kalman_classes.append(node_type_dataset.kalman_classes)
            self.class_count_dict.append(node_type_dataset.class_count_dict)
            self.class_weights.append(node_type_dataset.balanced_class_weights)
            self.inv_class_weights.append(node_type_dataset.class_weights)
            self.boarders.append(node_type_dataset.borders)

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
                 scene_freq_mult, hyperparams, augment=False, stack_right=0.007, borders=None, predifined_num_classes=None, **kwargs):
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
        self.num_classes = predifined_num_classes
        self.load_scores()
        # self.rebalance_bins_binary()
        # self.rebalance_bins_multi(stack_right=stack_right)
        self.rebalance_3_bins(borders=borders)

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

    def rebalance_bins_binary(self, split=0.1):
        env_name = self.env.scenes[0].name
        with open(os.path.join(self.scores_path, '%s_kalman.pkl' % env_name), 'rb') as f:
            scores = dill.load(f)
        lbls = (scores / 0.5).astype(np.int)
        # Calculating class values counts
        dic = {}
        for i in range(lbls.max() + 1):
            dic[i] = 0
        for l in lbls:
            dic[l] += 1
        # split point
        split_count = scores.shape[0] * (1 - split)
        cum_sum = 0
        for i in range(lbls.max() + 1):
            if cum_sum + dic[i] > split_count:
                split_cls = i  # included in majority
                break
            else:
                cum_sum = cum_sum + dic[i]
        lbls = np.where(lbls <= split_cls, 0, lbls)
        lbls = np.where(lbls > split_cls, 1, lbls)
        dic_ = {}
        for i in range(lbls.max() + 1):
            dic_[i] = 0
        for l in lbls:
            dic_[l] += 1
        class_count = [*dic_.values()]
        class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
        self.class_weights_all = class_weights[lbls]
        self.weighted_sampler = data.WeightedRandomSampler(
            weights=self.class_weights_all,
            num_samples=len(self.class_weights_all),
            replacement=True)
        n = scores.shape[0]
        beta = (n - 1) / n
        # import pdb; pdb.set_trace()
        self.balanced_class_weights = (1 - beta) / (1 - torch.pow(beta, torch.tensor(class_count, dtype=torch.float)))
        self.balanced_class_weights_all = self.balanced_class_weights[lbls]
        self.class_weights = class_weights
        self.kalman_classes = lbls
        self.class_count_dict = dic_

    def rebalance_bins_multi(self, stack_right):
        # TODO Use 1 spaced clusters
        env_name = self.env.scenes[0].name
        with open(os.path.join(self.scores_path, '%s_kalman.pkl' % env_name), 'rb') as f:
            scores = dill.load(f)
        lbls = (scores / 0.5).astype(np.int)

        # Calculating class values counts
        dic = {}
        for i in range(lbls.max() + 1):
            dic[i] = 0
        for l in lbls:
            dic[l] += 1
        # Stacking the right 0.7 percent into a class
        if self.num_classes is not None:
            minority_class = self.num_classes - 1
            for l in range(len(lbls)):
                if lbls[l] > minority_class:
                    lbls[l] = minority_class
            for i in range(minority_class + 1, lbls.max()):
                dic[minority_class] += dic[i]
                del(dic[i])
        elif stack_right is not None and isinstance(stack_right, float):
            dic_ = deepcopy(dic)
            sum_ = 0
            done = False
            i = lbls.max()
            verification_mask = (lbls == lbls.max())
            while i > 0 and not done:  # left 0.7 percent
                if sum_ + dic_[i] >= scores.shape[0] * stack_right:
                    done = True
                else:
                    sum_ += dic_[i]
                    del (dic_[i])
                    i -= 1
            dic_[i + 1] = sum_
            minority_class = i + 1
            for l in range(len(lbls)):
                if lbls[l] > minority_class:
                    lbls[l] = minority_class
            assert all(lbls[verification_mask] == minority_class)
            dic = dic_
        # Sorting classes lower has more data points
        original_keys = list(dic.keys())
        new_keys = sorted(original_keys, key=lambda x: dic[x], reverse=True)
        sorting_dic = {new_keys[k]: k for k in range(len(original_keys))}

        # Overwriting class values
        for l in range(len(lbls)):
            lbls[l] = sorting_dic[lbls[l]]

        # Calculating class values counts after sorting
        dic_sorted = {}
        for i in range(lbls.max() + 1):
            dic_sorted[i] = 0
        for l in lbls:
            dic_sorted[l] += 1
        assert sum(dic_sorted.values()) == scores.shape[0]

        # class weights .i.e sampling probability
        class_count = [*dic_sorted.values()]
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
        self.class_count_dict = dic_sorted

    def rebalance_3_bins(self, borders=None):
        # Borders : tuple(int, int) boarders
        # TODO Use 1 spaced clusters
        env_name = self.env.scenes[0].name
        with open(os.path.join(self.scores_path, '%s_kalman.pkl' % env_name), 'rb') as f:
            scores = dill.load(f)
        lbls = (scores / 0.5).astype(np.int)
        # Calculating class values counts
        dic = {}
        for i in range(lbls.max() + 1):
            dic[i] = 0
        for l in lbls:
            dic[l] += 1
        if borders == None:
            class_clusters = []
            borders = []
            # Stacking the right 0.7 percent into a class
            limits = [0.6, 0.95]
            cumsum = 0
            current_limit = 0
            current_list = []
            for c in dic.keys():
                if current_limit < 2 and cumsum + dic[c] >= scores.shape[0] * limits[current_limit]:
                    current_list.append(c)
                    class_clusters.append(current_list)  # incluse
                    borders.append(c)
                    cumsum += dic[c]
                    current_limit += 1
                    current_list = []
                elif c == lbls.max():
                    current_list.append(c)
                    class_clusters.append(current_list)  # incluse
                else:
                    cumsum += dic[c]
                    current_list.append(c)
            for c in range(3):
                lbls = np.where((lbls <= class_clusters[c][-1]) & (lbls >= class_clusters[c][0]), c, lbls)
        else:
            # the 2 borders are given
            class_clusters = []
            current_list = []
            current_limit = 0
            for c in dic.keys():
                if current_limit < 2 and c < borders[current_limit]:
                    current_list.append(c)
                elif current_limit == 2:
                    current_list.append(c)
                else:
                    current_list.append(c)
                    class_clusters.append(current_list)  # incluse
                    current_limit += 1
                    current_list = []
            class_clusters.append(current_list)  # incluse
            for c in range(3):
                lbls = np.where((lbls <= class_clusters[c][-1]) & (lbls >= class_clusters[c][0]), c, lbls)
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
        self.class_count_dict = dic_
        self.borders = borders

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
                                        self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.kalman_classes[i], self.scores[i])
        # scene = scene.augment()
        # node = scene.get_node_by_id(node.id)
        # sample_aug = get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
        #                                 self.edge_types, self.max_ht, self.max_ft, self.hyperparams) + (self.scores[i],)
        return sample
