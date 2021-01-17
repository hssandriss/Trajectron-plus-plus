import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron_multi import Trajectron
import evaluation
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data_train", help="full path to data file", type=str)
parser.add_argument("--data_val", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    #trajectron.set_annealing_params()
    return trajectron, hyperparams


if __name__ == "__main__":
    #######################################
    #### Training Data  Representation ####
    #######################################
    with open(args.data_train, 'rb') as f:
        env_train = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env_train, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env_train.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env_train.scenes

    print("-- Preparing Train Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env_train.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        
        ############### BEST OF 20 ###############
        train_features_list = []
        print("-- Evaluating Train best of 20")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Train Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions,features = eval_stg.predict(scene,timesteps,ph,num_samples=20,min_history_timesteps=7,min_future_timesteps=12,z_mode=False,gmm_mode=False,full_dist=False)
                train_features_list.append(features)

    train_features_list = [f for feature in train_features_list for f in feature if len(feature)!= 0] #some predictions are empty
    train_feat = torch.cat(train_features_list, dim = 0)

    #######################################
    ####     Val Data Representation   ####
    #######################################
    with open(args.data_val, 'rb') as f:
        env_val = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env_val, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env_val.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env_val.scenes

    print("-- Preparing Val Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env_val.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        
        ############### BEST OF 20 ###############
        val_features_list = []
        print("-- Evaluating Val best of 20")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Val Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                predictions,features = eval_stg.predict(scene,timesteps,ph,num_samples=20,min_history_timesteps=7,min_future_timesteps=12,z_mode=False,gmm_mode=False,full_dist=False)
                val_features_list.append(features)
                
    val_features_list = [f for feature in val_features_list for f in feature if len(feature)!= 0] #some predictions are empty
    val_feat = torch.cat(val_features_list, dim = 0)

    #######################################
    ####      TSNE Representation      ####
    #######################################
    tsne_input = torch.cat((train_feat, val_feat), dim = 0)
    idx_train_tsne = train_feat.shape[0]
    # train : tsne_input[:idx_train_tsne] , val: tsne_input[idx_train_tsne:]
    tsne_input = tsne_input.numpy()

    # metric: default is euclidean, 
    # check perplexity, early_exaggeration, learning_rate
    tsne_output = TSNE(n_components=2, init = 'pca', ).fit_transform(tsne_input)
    #tsne_output_normalized = normalize (tsne_output, axis = 0) # l2 normalization of each feature
    tsne_output_normalized = 2*((tsne_output - tsne_output.min(0)) / tsne_output.ptp(0)) -1

    plt.scatter(tsne_output_normalized[:idx_train_tsne,0], tsne_output_normalized[:idx_train_tsne,1], label='train')

    plt.scatter(tsne_output_normalized[idx_train_tsne:,0], tsne_output_normalized[idx_train_tsne:,1], label='val')
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.output_path, args.output_tag + '_normalized_features.png'))

    plt.cla()
    plt.scatter(tsne_output[:idx_train_tsne,0], tsne_output[:idx_train_tsne,1], label='train')
    plt.scatter(tsne_output[idx_train_tsne:,0], tsne_output[idx_train_tsne:,1], label='val')
    plt.legend(loc="best")
    plt.savefig(os.path.join(args.output_path, args.output_tag + '_features.png'))

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



