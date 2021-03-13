from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import torch.distributions.multivariate_normal as torchdist
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import dill
import argparse
import json
import math
import os
import random
import sys
from copy import deepcopy

sys.path.append("../../trajectron")

from utils import prediction_output_to_trajectories
from model.model_registrar import ModelRegistrar
from model.trajectron_multi import Trajectron

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--data2", help="full path to data file", type=str)
parser.add_argument("--model", help="path to model", type=str)
parser.add_argument("--checkpoint", help="checkpoint", type=int)
parser.add_argument("--chkpt_extra_tag", help="checkpoint extra tag", type=str, default=None)
parser.add_argument("--tagplot", help="tag for plot", type=str)
parser.add_argument("--save_output", type=str)

args = parser.parse_args()


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def rebalance_bins_binary(scores, border=None, split=0.1):
    lbls = (scores / 0.5).astype(np.int)
    # Calculating class values counts
    dic = {}
    for i in range(lbls.max() + 1):
        dic[i] = 0
    for l in lbls:
        dic[l] += 1
    if border is None:
        assert isinstance(split, float)
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
        border = split_cls
    else:
        assert isinstance(border, int)
        lbls = np.where(lbls <= border, 0, lbls)
        lbls = np.where(lbls > border, 1, lbls)
    dic_ = {}
    for i in range(lbls.max() + 1):
        dic_[i] = 0
    for l in lbls:
        dic_[l] += 1
    return lbls, dic_, border


def rebalance_3_bins(scores, borders=None):
    # Borders : tuple(int, int) boarders
    # TODO Use 1 spaced clusters
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

    return lbls, dic_, borders


def rebalance_bins(scores, stack_right=0.007, pred_num_classes=None):
    # TODO Use 1 spaced clusters
    lbls = (scores / 0.5).astype(np.int)

    # Calculating class values counts
    dic = {}
    for i in range(lbls.max() + 1):
        dic[i] = 0
    for l in lbls:
        dic[l] += 1
    import pdb; pdb.set_trace()
    # Stacking the right 0.7 percent into a class
    if pred_num_classes is not None and isinstance(pred_num_classes, int):
        minority_class = pred_num_classes - 1
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
    return lbls, dic_sorted


def load_model(model_dir, env, ts=100, extra_tag=None):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts, extra_tag)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    return trajectron, hyperparams


def get_kalman_filter_result(history):
    # history has shape [8, 2]
    z_x = history[:, 0]
    z_y = history[:, 1]

    # compute the average velocity over the history
    v_x = 0
    v_y = 0
    for l in range(history.shape[0] - 1):
        v_x += z_x[l + 1] - z_x[l]
        v_y += z_y[l + 1] - z_y[l]
    v_x = v_x / (history.shape[0] - 1)
    v_y = v_y / (history.shape[0] - 1)

    # the vector of location where we iterate to refine
    # 20 is the number of iterations (8 for history and 1 for future)
    x_x = np.zeros(9, np.float32)
    x_y = np.zeros(9, np.float32)

    # uncertainties of the location and scale
    P_x = np.zeros(9, np.float32)
    P_y = np.zeros(9, np.float32)

    # uncertainties of the velocity
    P_vx = np.zeros(9, np.float32)
    P_vy = np.zeros(9, np.float32)

    # we initialize the uncertainty to one (unit gaussian)
    P_x[0] = 1.0
    P_y[0] = 1.0

    P_vx[0] = 1.0
    P_vy[0] = 1.0

    x_x[0] = z_x[0]
    x_y[0] = z_y[0]

    # choose wisely Observation and Process Noise
    Q = 0.00001
    R = 0.0001

    K_x = np.zeros(9, np.float32)
    K_y = np.zeros(9, np.float32)

    K_vx = np.zeros(9, np.float32)
    K_vy = np.zeros(9, np.float32)

    # start iteration
    for k in range(history.shape[0] - 1):
        # print('iter: %d' % k)
        # predict
        x_x[k + 1] = x_x[k] + v_x
        x_y[k + 1] = x_y[k] + v_y
        P_x[k + 1] = P_x[k] + P_vx[k] + Q
        P_y[k + 1] = P_y[k] + P_vy[k] + Q
        P_vx[k + 1] = P_vx[k] + Q
        P_vy[k + 1] = P_vy[k] + Q

        # correct
        K_x[k + 1] = P_x[k + 1] / (P_x[k + 1] + R)
        K_y[k + 1] = P_y[k + 1] / (P_y[k + 1] + R)
        x_x[k + 1] = x_x[k + 1] + K_x[k + 1] * (z_x[k + 1] - x_x[k + 1])
        x_y[k + 1] = x_y[k + 1] + K_y[k + 1] * (z_y[k + 1] - x_y[k + 1])
        P_x[k + 1] = P_x[k + 1] - K_x[k + 1] * P_x[k + 1]
        P_y[k + 1] = P_y[k + 1] - K_y[k + 1] * P_y[k + 1]
        K_vx[k + 1] = P_vx[k + 1] / (P_vx[k + 1] + R)
        K_vy[k + 1] = P_vy[k + 1] / (P_vy[k + 1] + R)
        P_vx[k + 1] = P_vx[k + 1] - K_vx[k + 1] * P_vx[k + 1]
        P_vy[k + 1] = P_vy[k + 1] - K_vy[k + 1] * P_vy[k + 1]

    k = k + 1
    # print('iter: %d' % k)
    # predict into the future
    x_x[k + 1] = x_x[k] + v_x * 12
    x_y[k + 1] = x_y[k] + v_y * 12
    P_x[k + 1] = P_x[k] + P_vx[k] * 12 * 12 + Q
    P_y[k + 1] = P_y[k] + P_vy[k] * 12 * 12 + Q
    P_vx[k + 1] = P_vx[k] + Q
    P_vy[k + 1] = P_vy[k] + Q

    z_future = np.zeros([2])
    z_future[0] = x_x[k + 1]
    z_future[1] = x_y[k + 1]
    return z_future


def calculate_epe(pred, gt):
    diff_x = (gt[0] - pred[0]) * (gt[0] - pred[0])
    diff_y = (gt[1] - pred[1]) * (gt[1] - pred[1])
    epe = math.sqrt(diff_x + diff_y)
    return epe


def get_colors(n, n_start=0):
    if n <= 13:
        list_colors = ['red', 'green', 'blue', 'purple', 'navy', 'brown',
                       'yellow', 'black', 'orange', 'pink', 'cyan', 'grey', 'lightgreen']
        colors = list_colors[n_start:n + 1]
    else:
        colors = [(np.random.choice(range(256)), np.random.choice(range(256)), np.random.choice(range(256)))
                  for i in range(n + 1 - n_start)]
    return colors


if __name__ == "__main__":

    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint, extra_tag=args.chkpt_extra_tag)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)
    scenes = env.scenes

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    # if args.data2:

    #     with open(args.data2, 'rb') as f:
    #         env2 = dill.load(f, encoding='latin1')
    #     eval_stg2, hyperparams2 = load_model(args.model, env2, ts=args.checkpoint, extra_tag=args.chkpt_extra_tag)

    #     if 'override_attention_radius' in hyperparams2:
    #         for attention_radius_override in hyperparams2['override_attention_radius']:
    #             node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
    #             env2.attention_radius[(node_type1, node_type2)] = float(attention_radius)
    #     scenes2 = env2.scenes

    #     print("-- Preparing Node Graph")
    #     for scene in tqdm(scenes2):
    #         scene.calculate_scene_graph(env2.attention_radius,
    #                                     hyperparams2['edge_addition_filter'],
    #                                     hyperparams2['edge_removal_filter'])
    #     ph2 = hyperparams2['prediction_horizon']
    #     max_hl2 = hyperparams2['maximum_history_length']

    with torch.no_grad():
        epes = []
        features_list = []
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            timesteps = np.arange(scene.timesteps)
            predictions, features = eval_stg.predict(scene,
                                                     timesteps,
                                                     ph,
                                                     min_history_timesteps=7,  # if 'test' in args.data else 1,
                                                     min_future_timesteps=12)
            (prediction_dict, histories_dict, futures_dict) = prediction_output_to_trajectories(predictions,
                                                                                                scene.dt,
                                                                                                max_hl,
                                                                                                ph,
                                                                                                prune_ph_to_future=True)

            for t in prediction_dict.keys():
                for node in prediction_dict[t].keys():
                    z_future = get_kalman_filter_result(histories_dict[t][node])
                    epe = calculate_epe(z_future, futures_dict[t][node][-1, :])
                    epes.append(epe)
            features_list.append(features)
        feat = torch.cat([features_list[i][0] for i in range(len(features_list))], dim=0)

        kalman_errors = np.array(epes)
        print('Kalman (FDE): %.2f' % (np.mean(kalman_errors)))
        assert feat.shape[0] == kalman_errors.shape[0]
        # kalman_classes, class_count_dict = rebalance_bins(kalman_errors, stack_right=0.007, pred_num_classes=None)
        # kalman_classes, class_count_dict, borders = rebalance_3_bins(kalman_errors)
        kalman_classes, class_count_dict, border = rebalance_bins_binary(kalman_errors)
        assert kalman_classes.shape[0] == feat.shape[0]
        num_classes = len(class_count_dict)
        # Subsampling from the training data

        min_nsamples = min([class_count_dict[k] for k in range(num_classes)])
        # min_nsamples = 1000
        total_sampled_indexes = []
        for c in range(num_classes):
            idx_c = np.where(kalman_classes == c)[0]
            np.random.shuffle(idx_c)
            total_sampled_indexes.append(idx_c[:min_nsamples])
        total_sampled_indexes = np.concatenate(total_sampled_indexes)
        np.random.shuffle(total_sampled_indexes)
        feat = feat[total_sampled_indexes]
        kalman_classes = kalman_classes[total_sampled_indexes]
        print()
        # import pdb; pdb.set_trace()
        # if args.data2:
        #     epes = []
        #     features_list = []
        #     for i, scene in enumerate(scenes2):
        #         print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
        #         timesteps = np.arange(scene.timesteps)
        #         predictions, features = eval_stg.predict(scene,
        #                                                  timesteps,
        #                                                  ph2,
        #                                                  min_history_timesteps=7,  # if 'test' in args.data else 1,
        #                                                  min_future_timesteps=12)
        #         (prediction_dict, histories_dict, futures_dict) = prediction_output_to_trajectories(predictions,
        #                                                                                             scene.dt,
        #                                                                                             max_hl2,
        #                                                                                             ph2,
        #                                                                                             prune_ph_to_future=True)
        #         for t in prediction_dict.keys():
        #             for node in prediction_dict[t].keys():
        #                 z_future = get_kalman_filter_result(histories_dict[t][node])
        #                 epe = calculate_epe(z_future, futures_dict[t][node][-1, :])
        #                 epes.append(epe)
        #         features_list.append(features)
        #     feat2 = torch.cat([features_list[i][0] for i in range(len(features_list))], dim=0)
        #     kalman_errors2 = np.array(epes)
        #     print('Kalman (FDE): %.2f' % (np.mean(kalman_errors2)))
        #     assert feat2.shape[0] == kalman_errors2.shape[0]
        #     import pdb; pdb.set_trace()
        #     # kalman_classes2, _ = rebalance_bins(kalman_errors, stack_right=None, pred_num_classes=num_classes)
        #     kalman_classes2, _, _ = rebalance_3_bins(kalman_errors2, borders=borders)
        #     # kalman_classes2, _, _ = rebalance_bins_binary(kalman_errors, border=border)
    tsne_input = feat.numpy()
    print('---------- Start TSNE ----------')
    tsne_output = TSNE(n_components=2, init='pca').fit_transform(tsne_input)
    # tsne_output_normalized = normalize (tsne_output, axis = 0) # l2 normalization of each feature
    tsne_output_normalized = 2 * ((tsne_output - tsne_output.min(0)) / tsne_output.ptp(0)) - 1

    # tsne_output_train = tsne_output[: train_idx]
    # tsne_output_test = tsne_output[train_idx:]

    #tsne_output_normalized_train = tsne_output_normalized[: train_idx]
    #tsne_output_normalized_test = tsne_output_normalized[train_idx:]
    print("---------- Saving Plots ---------- ")

    colors_train = get_colors(kalman_classes.max())
    # import pdb; pdb.set_trace()
    #colors_test = get_colors(n = kalman_classes_balanced.max()+ kalman_classes_balanced_test.max()+1, n_start= kalman_classes_balanced.max() + 1)
    fig = plt.figure(figsize=(8, 8))

    for i in range(kalman_classes.max() + 1):
        output_c = tsne_output[np.where(kalman_classes == i)[0]]
        plt.scatter(output_c[:, 0], output_c[:, 1], color=colors_train[i], label='train_class_' + str(i))
        plt.legend()
        plt.savefig(os.path.join(args.model, args.tagplot + '.png'))
        #import pdb; pdb.set_trace()
    for label in range(kalman_classes.max() + 1):
        idx_label_train = np.where(kalman_classes == label)[0]
        #idx_label_test =  np.where(kalman_classes_balanced_test == label)[0]
        tsne_output_train_label = tsne_output[np.where(kalman_classes == label)[0]]
        #tsne_output_test_label = tsne_output_test[np.where(kalman_classes_balanced_test==label)[0]]
        plt.clf()
        plt.scatter(tsne_output_train_label[:, 0], tsne_output_train_label[:, 1], c=colors_train[label], label='train')
        #plt.scatter(tsne_output_test_label[:,0], tsne_output_test_label[:,1], color = colors_test[label], label = 'test')

        plt.legend()
        plt.savefig(os.path.join(args.model, args.tagplot + '_class_' + str(label) + '.png'))

    #######################################
    ####      TSNE Representation      ####
    #######################################
    # print('---------- Start TSNE ----------')
    # tsne_input = feat
    # tsne_input = tsne_input.numpy()
    # tsne_output = TSNE(n_components=2, init='pca').fit_transform(tsne_input)
    # tsne_output_normalized = 2 * ((tsne_output - tsne_output.min(0)) / tsne_output.ptp(0)) - 1
    # # if args.data2:
    # #     tsne_input2 = feat2
    # #     tsne_input2 = tsne_input2.numpy()
    # #     tsne_output2 = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(tsne_input2)
    # #     tsne_output_normalized2 = 2 * ((tsne_output - tsne_output.min(0)) / tsne_output.ptp(0)) - 1

    print("---------- Saving Plots ---------- ")
    N = num_classes
    # define the colormap
    cmap = plt.cm.jet
    # extract all colors from the .jet map
    # cmaplist = [cmap(i) for i in range(0, cmap.N, cmap.N//N)]
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds, step = np.linspace(0, N, N + 1, retstep=True)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    fig = plt.figure(figsize=(10, 8))
    plt.scatter(tsne_output[:, 0], tsne_output[:, 1], c=kalman_classes, cmap=cmap, norm=norm)
    cb = plt.colorbar(spacing='proportional', ticks=bounds)
    cb.set_label('Kalman classes')
    figname = os.path.join(args.model, args.tagplot + '_2.png')
    plt.savefig(figname)

    labels = [i for i in range(num_classes)]
    for label in labels:
        idx_label = np.where(kalman_classes == label)[0]
        tsne_output_label = tsne_output[idx_label, :]
        plt.clf()
        plt.scatter(tsne_output_label[:, 0], tsne_output_label[:, 1], c=np.array(
            [label for _ in range(len(idx_label))]), cmap=cmap, norm=norm)
        figname = os.path.join(args.model, args.tagplot + '_class_' + str(label) + '.png')
        plt.savefig(figname)

    # if args.data2:
    #     plt.clf()
    #     fig = plt.figure(figsize=(8, 8))
    #     plt.scatter(tsne_output2[:, 0], tsne_output2[:, 1], c=kalman_classes2, cmap=cmap, norm=norm)
    #     cb = plt.colorbar(spacing='proportional', ticks=bounds)
    #     cb.set_label('Kalman classes')
    #     figname = os.path.join(args.model, args.tagplot + '_data2.png')
    #     plt.savefig(figname)
    #     for label in labels:
    #         idx_label = np.where(kalman_classes2 == label)[0]
    #         tsne_output_label = tsne_output2[idx_label, :]
    #         plt.clf()
    #         plt.scatter(tsne_output_label[:, 0], tsne_output_label[:, 1], c=np.array(
    #             [label for _ in range(len(idx_label))]), cmap=cmap, norm=norm)
    #         figname = os.path.join(args.model, args.tagplot + '_class_' + str(label) + '_data2.png')
    #         plt.savefig(figname)
