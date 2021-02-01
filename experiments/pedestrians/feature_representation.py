import argparse
import json
import math
import os
import random
import sys
from copy import deepcopy

import dill
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("../../trajectron")
from model.model_registrar import ModelRegistrar
from model.trajectron_multi import Trajectron
from utils import prediction_output_to_trajectories

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
parser.add_argument("--model", help="path to model", type=str)
parser.add_argument("--checkpoint", help="checkpoint", type=int)
parser.add_argument("--chkpt_extra_tag", help="checkpoint extra tag", type=str, default=None)
parser.add_argument("--tagplot", help="tag for plot", type=str)
parser.add_argument("--save_output", type=str)

args = parser.parse_args()


def rebalance_bins(scores):
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
        if sum_ + dic_[i] >= scores.shape[0]*0.007:
            done = True
        else:
            sum_ += dic_[i]
            del (dic_[i])
            i -= 1
    dic_[i+1] = sum_

    original_keys = dic_.keys()
    original_keys = list(original_keys)
    new_keys = sorted(original_keys, key=lambda x: dic_[x], reverse=True)
    switched_dic = {new_keys[k]: k for k in range(len(original_keys))}
    minority_class = i+1
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
    kalman_classes = lbls
    class_count_dict = dic_compare
    return kalman_classes, class_count_dict


def load_model(model_dir, env, ts=100, extra_tag=None):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts, extra_tag )
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
    P_x[k + 1] = P_x[k] + P_vx[k] * 12*12 + Q
    P_y[k + 1] = P_y[k] + P_vy[k] * 12*12 + Q
    P_vx[k + 1] = P_vx[k] + Q
    P_vy[k + 1] = P_vy[k] + Q

    z_future = np.zeros([2])
    z_future[0] = x_x[k + 1]
    z_future[1] = x_y[k + 1]
    return z_future


def calculate_epe(pred, gt):
    diff_x = (gt[0] - pred[0]) * (gt[0] - pred[0])
    diff_y = (gt[1] - pred[1]) * (gt[1] - pred[1])
    epe = math.sqrt(diff_x+diff_y)
    return epe


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

        kalman_classes, class_count_dict = rebalance_bins(kalman_errors)
        num_classes = len(class_count_dict)

        import pdb; pdb.set_trace()
        #######################################
        ####      TSNE Representation      ####
        #######################################
        tsne_input = feat
        tsne_input = tsne_input.numpy()

        # metric: default is euclidean,
        # check perplexity, early_exaggeration, learning_rate
        print('---------- Start TSNE ----------')
        tsne_output = TSNE(n_components=2, init='pca', ).fit_transform(tsne_input)
        # tsne_output_normalized = normalize (tsne_output, axis = 0) # l2 normalization of each feature
        tsne_output_normalized = 2*((tsne_output - tsne_output.min(0)) / tsne_output.ptp(0)) - 1

        #colors = [(random.random(),random.random(),random.random()) for i in range(kalman_classes.max()+ 1) ]
        # list(matplotlib.colors.cnames.keys())
        
        all_colors = [color for color in list(matplotlib.colors.cnames.keys()) if 'white' not in str(color)]
        colors = random.sample(all_colors, num_classes)
        # colors = ['red', 'green', 'blue', 'purple', 'navy', 'brown', 'yellow', 'black', 'orange', 'pink', 'cyan', 'grey', 'lightgreen']
        fig = plt.figure(figsize=(8, 8))
        plt.scatter(tsne_output[:, 0], tsne_output[:, 1], c=kalman_classes, cmap=matplotlib.colors.ListedColormap(colors))
        labels = [i for i in range(kalman_classes.max()+1)]
        cb = plt.colorbar()
        loc = np.arange(0, max(kalman_classes), max(kalman_classes)/float(len(colors)))
        cb.set_ticks(loc)
        cb.set_ticklabels(labels)
        figname = os.path.join(args.model, args.tagplot + '.png')
        plt.savefig(figname)
        for label in labels:
            idx_label = np.where(kalman_classes == label)[0]
            tsne_output_label = tsne_output[idx_label, :]
            plt.clf()
            plt.scatter(tsne_output_label[:, 0], tsne_output_label[:, 1], c=colors[label])
            figname = os.path.join(args.model, args.tagplot + '_class_' + str(label)+'.png')
            plt.savefig(figname)

        import pdb; pdb.set_trace()
        # plt.scatter(tsne_output_normalized[:idx_train_tsne,0], tsne_output_normalized[:idx_train_tsne,1], label='train')
        # plt.scatter(tsne_output_normalized[idx_train_tsne:,0], tsne_output_normalized[idx_train_tsne:,1], label='val')
        # plt.legend(loc="best")
        # plt.savefig(os.path.join(args.output_path, args.output_tag + '_normalized_features.png'))
        # plt.cla()
        # plt.scatter(tsne_output[:idx_train_tsne,0], tsne_output[:idx_train_tsne,1], label='train')
        # plt.scatter(tsne_output[idx_train_tsne:,0], tsne_output[idx_train_tsne:,1], label='val')
        # plt.legend(loc="best")
        # plt.savefig(os.path.join(args.output_path, args.output_tag + '_features.png'))
        ########### 3D TSNE ##########
        # tsne_output3D = TSNE(n_components=3, init='pca', ).fit_transform(tsne_input)
        # tsne_output_normalized3D = 2*((tsne_output3D - tsne_output3D.min(0)) / tsne_output3D.ptp(0)) - 1
        # plt.cla()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(tsne_output3D[:idx_train_tsne, 0], tsne_output3D[:idx_train_tsne, 1], -tsne_output3D[:idx_train_tsne, 2], zdir='z', label='train')
        # ax.scatter(tsne_output3D[idx_train_tsne:, 0], tsne_output3D[idx_train_tsne:, 1], -tsne_output3D[idx_train_tsne:, 2], zdir='z', label='val')
        # plt.legend(loc="best")
        # plt.savefig(os.path.join(args.output_path, args.output_tag + '_3D_features.png'))
        # plt.cla()
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(tsne_output_normalized3D[:idx_train_tsne, 0], tsne_output_normalized3D[:idx_train_tsne, 1], -tsne_output_normalized3D[:idx_train_tsne, 2], zdir='z', label='train')
        # ax.scatter(tsne_output_normalized3D[idx_train_tsne:, 0], tsne_output_normalized3D[idx_train_tsne:, 1], -tsne_output_normalized3D[idx_train_tsne:, 2], zdir='z', label='val')
        # plt.legend(loc="best")
        # plt.savefig(os.path.join(args.output_path, args.output_tag + '_normalized3D_features.png'))
