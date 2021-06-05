import argparse
import json
import os
import random
import sys

import dill
import numpy as np
import pandas as pd
import torch

sys.path.append("../../trajectron")
import argparse
import copy
import glob
import json
import math
import os
import pickle
import sys
from copy import deepcopy

import dill
import evaluation
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributions.multivariate_normal as torchdist
from matplotlib import pyplot as plt
from model.model_registrar import ModelRegistrar
from model.trajectron_multi import Trajectron
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append("../../trajectron")
import evaluation
from model.model_registrar import ModelRegistrar
from tqdm import tqdm
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
parser.add_argument("--data_test", help="full path to data file", type=str)
parser.add_argument("--model", help="path to model", type=str)
parser.add_argument("--checkpoint", help="checkpoint", type=int)
parser.add_argument("--tagplot", help="tag for plot", type=str)
parser.add_argument("--save_output", type=str)

args = parser.parse_args()

def rebalance_bins_train(scores, binary = False):
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
    if binary:
        switched_dic = {}
        while i > 0 and not done: # left 10 percent
            if sum_ + dic_[i] >= scores.shape[0]*0.10:
                done = True
            else:
                sum_ += dic_[i]
                del (dic_[i])
                switched_dic[i] = 1
                i -=1 
        sum_1 = sum_
        sum_0 = 0
        while i >0:
            sum_0 += dic_[i]
            del (dic_[i])
            switched_dic[i] = 0
            i -=1
        switched_dic[0] = 0
        dic_[0] += sum_0
        dic_[1] = sum_1
        assert sum(dic_.values()) == scores.shape[0]

        for l in range(len(lbls)):
            lbls[l] = switched_dic[lbls[l]]
        
        dic_compare = {}
        for i in range(lbls.max() + 1):
            dic_compare[i] = 0
        for l in lbls:
            dic_compare[l] += 1 
    else:
        while i > 0 and not done: # left 0.7 percent
            if sum_ + dic_[i] >= scores.shape[0]*0.007:
                done = True
            else:
                sum_ += dic_[i]
                del (dic_[i])
                i -=1 
        dic_[i+1] = sum_

        original_keys = dic_.keys()
        original_keys = list(original_keys)
        new_keys = sorted(original_keys, key = lambda x: dic_[x], reverse = True)
        switched_dic = {new_keys[k]:k for k in range(len(original_keys))}
        minority_class = i+1
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
    assert sum(dic_.values()) == scores.shape[0]
    class_count = [*dic_.values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    kalman_classes = lbls
    class_count_dict = dic_compare
    return kalman_classes, switched_dic

def rebalance_bins_test(scores, switched_dic, binary = True):
    import pdb; pdb.set_trace()
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
    import pdb; pdb.set_trace()
    if binary:
        for l in range(len(lbls)):
            lbls[l] = switched_dic[lbls[l]]
        import pdb; pdb.set_trace()
        dic_compare = {}
        for i in range(lbls.max() + 1):
            dic_compare[i] = 0
        for l in lbls:
            dic_compare[l] += 1 
        import pdb; pdb.set_trace()
    else:
        while i > 0 and not done: # left 0.7 percent
            if sum_ + dic_[i] >= scores.shape[0]*0.007:
                done = True
            else:
                sum_ += dic_[i]
                del (dic_[i])
                i -=1 
        dic_[i+1] = sum_

        original_keys = dic_.keys()
        original_keys = list(original_keys)
        new_keys = sorted(original_keys, key = lambda x: dic_[x], reverse = True)
        switched_dic = {new_keys[k]:k for k in range(len(original_keys))}
        minority_class = i+1
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
    assert sum(dic_.values()) == scores.shape[0]
    class_count = [*dic_.values()]
    class_weights = 1. / torch.tensor(class_count, dtype=torch.float)
    kalman_classes = lbls
    class_count_dict = dic_compare
    return kalman_classes


def rebalance_3_bins( scores, borders = None):
    lbls = (scores / 0.5).astype(np.int)
    # Calculating class values counts
    dic = {}
    for i in range(lbls.max() + 1):
        dic[i] = 0
    for l in lbls:
        dic[l] += 1
    lbls_all = lbls.copy()
    if borders == None:
        # train dataset
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

    kalman_classes = lbls
    return kalman_classes, borders, lbls_all

def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
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

def get_feat_balanced(feat, kalman_classes, portion = 1000):
    final_index= np.empty([1], dtype=int)
    for c in range(max(kalman_classes)+1):
        idx_c = np.where(kalman_classes==c)[0]
        if portion != 0:
            np.random.shuffle(idx_c)
            idx = idx_c[:portion]
            final_index = np.concatenate((final_index, idx), axis = 0)
    final_index = np.delete(final_index, [0])
    np.random.shuffle(final_index)
    feat_balanced = feat[final_index]
    kalman_classes_balanced = kalman_classes[final_index] 
    return feat_balanced, kalman_classes_balanced

def get_colors(n, n_start = 0):
    if n <= 13:
        list_colors = ['red','green','blue','purple', 'navy', 'brown', 'yellow', 'black', 'orange', 'pink', 'cyan', 'grey', 'lightgreen']
        colors = list_colors[n_start:n+1]
    else:
        colors = [(np.random.choice(range(256)), np.random.choice(range(256)), np.random.choice(range(256))) for i in range(n+1- n_start)]
    return colors

if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')
    
    with open(args.data_test, 'rb') as f:
        env_test = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    
    scenes = env.scenes
    scenes_test = env_test.scenes

    print("-- Preparing Node Train Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])
    
    print("-- Preparing Node Test Graph")
    for scene in tqdm(scenes_test):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        ############# Training set ############
        epes = []
        features_list = []
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            timesteps = np.arange(scene.timesteps)
            predictions, features = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           min_history_timesteps=7, #if 'test' in args.data else 1,
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

        ############# Test set ############
        epes_test = []
        features_list_test = []
        for i, scene in enumerate(scenes_test):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes_test)}")
            timesteps = np.arange(scene.timesteps)
            predictions, features = eval_stg.predict(scene,
                                           timesteps,
                                           ph,
                                           min_history_timesteps=7, #if 'test' in args.data else 1,
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
                    epes_test.append(epe)
            features_list_test.append(features)
        feat_test = torch.cat([features_list_test[i][0] for i in range(len(features_list_test))], dim=0)

        kalman_errors_test = np.array(epes_test)
        print('Kalman Test (FDE): %.2f' % (np.mean(kalman_errors_test)))


        assert feat.shape[0] == kalman_errors.shape[0]
        assert feat_test.shape[0] == kalman_errors_test.shape[0]
        #with open(args.save_output + '.pkl', 'wb') as f_writer:
        #    dill.dump(kalman_errors, f_writer)
        #kalman_classes, switched_dic = rebalance_bins_train(kalman_errors)
        kalman_classes, borders_train, kalman_classes_all = rebalance_3_bins(kalman_errors)
        kalman_classes_test, _, kalman_classes_all_test = rebalance_3_bins(kalman_errors_test, borders_train)
        #kalman_classes_test = rebalance_bins_test(kalman_errors_test, switched_dic)

        #######################################
        ####      TSNE Representation      ####
        #######################################
        feat_balanced, kalman_classes_balanced = get_feat_balanced(feat, kalman_classes)
        feat_balanced_all, kalman_classes_balanced_all = get_feat_balanced(feat, kalman_classes_all, 100)
        feat_balanced_all_test, kalman_classes_balanced_all_test = get_feat_balanced(feat_test, kalman_classes_all_test, 100)
        
        #feat_balanced_test, kalman_classes_balanced_test = get_feat_balanced(feat, kalman_classes_test)
        #tsne_input = torch.cat((train_feat, val_feat), dim = 0)
        train_idx = len(feat_balanced)
        tsne_input = feat_balanced
        #tsne_input = np.concatenate((feat_balanced, feat_balanced_test), axis = 0)
        print('---------- histograms ----------')
        fig = plt.figure(figsize=(8,8))
        plt.hist(kalman_classes_all, bins = np.arange(np.amax(kalman_classes_all)))
        plt.savefig(os.path.join(args.model ,args.tagplot+ '_histogramTrain'+'.png'))
        fig = plt.figure(figsize=(8,8))
        plt.hist(kalman_classes_all_test, bins = np.arange(np.amax(kalman_classes_all_test)))
        plt.savefig(os.path.join(args.model ,args.tagplot+ '_histogramTest'+'.png'))
        
        print('---------- Weights analysis ----------')
        weights_to_plot = [ 'model_dict.PEDESTRIAN/decoder/rnn_cell.weight_ih', 'model_dict.PEDESTRIAN/decoder/rnn_cell.weight_hh' ,'model_dict.PEDESTRIAN/decoder/initial_h.weight', 'model_dict.PEDESTRIAN/decoder/initial_mu.weight',  'model_dict.PEDESTRIAN/decoder/proj_to_mus.weight']
        weights_name = [ 'rnn_cell_ih', 'rnn_cell_hh', 'initial_h', 'initial_mu',  'proj_to_mus']
        for i in range(len(weights_to_plot)):
            import pdb; pdb.set_trace()
            current_weight = eval_stg.model_registrar.state_dict()[weights_to_plot[i]].numpy()
            fig = plt.figure(figsize=(20,20))
            plt.matshow(current_weight)
            plt.savefig(os.path.join(args.model ,args.tagplot+ '_WEIGHTS_'+ weights_name[i]+ '.png'))
            fig = plt.figure(figsize=(20,20))
            plt.hist(current_weight)
            plt.savefig(os.path.join(args.model ,args.tagplot+ '_WEIGHTS_HIST_'+ weights_name[i]+ '.png'))
        
        # metric: default is euclidean, 
        # check perplexity, early_exaggeration, learning_rate
        print('---------- Start TSNE TRAIN ----------')
        tsne_output = TSNE(n_components=2, init = 'pca').fit_transform(tsne_input)
        #tsne_output_normalized = normalize (tsne_output, axis = 0) # l2 normalization of each feature
        tsne_output_normalized = 2*((tsne_output - tsne_output.min(0)) / tsne_output.ptp(0)) -1

        tsne_output_train = tsne_output[: train_idx]
        tsne_output_test = tsne_output[train_idx:]

        #tsne_output_normalized_train = tsne_output_normalized[: train_idx]
        #tsne_output_normalized_test = tsne_output_normalized[train_idx:]
        
        colors_train = get_colors(kalman_classes_balanced.max())
        #colors_test = get_colors(n = kalman_classes_balanced.max()+ kalman_classes_balanced_test.max()+1, n_start= kalman_classes_balanced.max() + 1)
        fig = plt.figure(figsize=(8,8))
        
        for i in range(kalman_classes_balanced.max()+1):
            output_c = tsne_output_train[np.where(kalman_classes_balanced==i)[0]]
            plt.scatter(output_c[:,0], output_c[:,1], color = colors_train[i], label = 'train_class_'+ str(i))
        #plt.scatter(tsne_output_train_maj[:,0], tsne_output_train_maj[:,1], color = colors_train[1], label = 'train_majority')

        plt.legend()
        plt.savefig(os.path.join(args.model, args.tagplot+'.png'))
        #import pdb; pdb.set_trace()
        for label in range(kalman_classes_balanced.max()+1):
            idx_label_train =  np.where(kalman_classes_balanced == label)[0]
            #idx_label_test =  np.where(kalman_classes_balanced_test == label)[0]
            tsne_output_train_label = tsne_output_train[np.where(kalman_classes_balanced==label)[0]]
            #tsne_output_test_label = tsne_output_test[np.where(kalman_classes_balanced_test==label)[0]]
            plt.clf()
            plt.scatter(tsne_output_train_label[:,0], tsne_output_train_label[:,1] , c = colors_train[label], label = 'train')
            #plt.scatter(tsne_output_test_label[:,0], tsne_output_test_label[:,1], color = colors_test[label], label = 'test')

            plt.legend()
            plt.savefig(os.path.join(args.model ,args.tagplot+ '_class_'+ str(label)+'.png'))
        
        
        tsne_input = feat_balanced_all
        train_idx = len(tsne_input)
        tsne_output = TSNE(n_components=2, init = 'pca').fit_transform(tsne_input)
        #tsne_output_normalized = normalize (tsne_output, axis = 0) # l2 normalization of each feature
        tsne_output_normalized = 2*((tsne_output - tsne_output.min(0)) / tsne_output.ptp(0)) -1

        tsne_output_train = tsne_output[: train_idx]
        tsne_output_test = tsne_output[train_idx:]

        plt.cla()
        plt.scatter(tsne_output_train[:,0], tsne_output_train[:,1], c=kalman_classes_balanced_all, cmap='viridis_r')
        plt.colorbar()
        #plt.legend()
        plt.savefig(os.path.join(args.model ,args.tagplot+ '_train_all'+'.png'))
        print('---------- Start TSNE Test ----------')
        fig = plt.figure(figsize=(8,8))
        tsne_input = feat_balanced_all_test
        train_idx = len(tsne_input)
        tsne_output_test = TSNE(n_components=2, init = 'pca').fit_transform(tsne_input)
        #tsne_output_normalized = normalize (tsne_output, axis = 0) # l2 normalization of each feature
        tsne_output_normalized_test = 2*((tsne_output_test - tsne_output_test.min(0)) / tsne_output_test.ptp(0)) -1

       

        plt.cla()
        plt.scatter(tsne_output_test[:,0], tsne_output_test[:,1], c=kalman_classes_balanced_all_test, cmap='viridis_r')
        plt.colorbar()
        #plt.legend()
        plt.savefig(os.path.join(args.model ,args.tagplot+ '_test_all'+'.png'))
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
        tsne_output3D = TSNE(n_components=3, init = 'pca', ).fit_transform(tsne_input)
        tsne_output_normalized3D = 2*((tsne_output3D - tsne_output3D.min(0)) / tsne_output3D.ptp(0)) -1

        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_output3D[:idx_train_tsne,0], tsne_output3D[:idx_train_tsne,1], -tsne_output3D[:idx_train_tsne,2], zdir='z', label= 'train')
        ax.scatter(tsne_output3D[idx_train_tsne:, 0], tsne_output3D[idx_train_tsne:,1], -tsne_output3D[idx_train_tsne:,2], zdir='z', label= 'val')
        plt.legend(loc="best")
        plt.savefig(os.path.join(args.output_path, args.output_tag + '_3D_features.png'))

        plt.cla()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(tsne_output_normalized3D[:idx_train_tsne,0], tsne_output_normalized3D[:idx_train_tsne,1], -tsne_output_normalized3D[:idx_train_tsne,2], zdir='z', label= 'train')
        ax.scatter(tsne_output_normalized3D[idx_train_tsne:, 0], tsne_output_normalized3D[idx_train_tsne:,1], -tsne_output_normalized3D[idx_train_tsne:,2], zdir='z', label= 'val')
        plt.legend(loc="best")
        plt.savefig(os.path.join(args.output_path, args.output_tag + '_normalized3D_features.png'))
