import pickle
import numpy as np
import os
import math
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
import copy
import matplotlib.pyplot as plt
import dill
import json
import argparse
sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
from utils import prediction_output_to_trajectories
import evaluation

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--save_output", type=str)

args = parser.parse_args()

def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
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

    eval_stg, hyperparams = load_model('models/models_03_Aug_2020_18_07_44_univ_ewta', env, ts=90)

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
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            timesteps = np.arange(scene.timesteps)
            predictions, _ = eval_stg.predict(scene,
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

        kalman_errors = np.array(epes)
        print('Kalman (FDE): %.2f' % (np.mean(kalman_errors)))
        with open(args.save_output + '.pkl', 'wb') as f_writer:
            dill.dump(kalman_errors, f_writer)