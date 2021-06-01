import argparse
import json
import math
import os
import sys

import dill
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append("../../trajectron")

import evaluation
from model.model_registrar import ModelRegistrar
from utils import prediction_output_to_trajectories

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument("--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--ldam", help="ldam or not", type=str)
parser.add_argument("--scores_dir", help="scores path", type=str)
args = parser.parse_args()


def load_model(model_dir, env, ldam, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    hyperparams['output_con_model'] = 64

    if ldam == "yes":
        class_count_dict = [{0: 25130, 1: 10805, 2: 1861}]
        borders = [2, 6]
        # [[0, 1, 2], [3, 4, 5, 6], [7 .. 32]]
        trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu', class_count_dict=class_count_dict)
    else:
        class_count_dict = [{0: 25130, 1: 10805, 2: 1861}]
        borders = [2, 6]
        trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    # trajectron.set_annealing_params()
    return trajectron, hyperparams, borders


def get_class_label(epe, borders, nb_classes):
    '''
    nb_classes = 3
    bordes = [2,6] 
    [0,1,2] ==> 0 ; [3,4,5, 6] ==> 1 ; [7 ...] ==> 2 
    '''
    label = None
    for i in range(len(borders)):
        if borders[i] > epe:
            label = i
            break
    if label == None:
        label = nb_classes - 1
    return label


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


def get_eval_percent(eval_ade_batch_errors, eval_fde_batch_errors, eval_kde_nll, percent_idx, current_epe_list, predictions):
    if len(percent_idx) == 0:
        return eval_ade_batch_errors, eval_fde_batch_errors, eval_kde_nll
    else:
        predictions_new = {}
        curr_iter = 0
        curr_iter_percent_idx = 0
        done = 0
        while curr_iter_percent_idx < len(percent_idx):
            for k, v in predictions.items():
                if isinstance(v, dict):
                    for k1, v1 in v.items():
                        if curr_iter == percent_idx[curr_iter_percent_idx]:
                            if k in predictions_new:
                                pass
                            else:
                                predictions_new[k] = {}
                            predictions_new[k][k1] = v1
                            done += 1
                            curr_iter_percent_idx += 1
                        if curr_iter_percent_idx == len(percent_idx):
                            break
                        curr_iter += 1

                else:
                    if curr_iter == percent_idx[curr_iter_percent_idx]:
                        if k in predictions_new:
                            pass
                        else:
                            predictions_new[k] = {}
                        predictions_new[k][k1] = v1
                        done += 1
                        curr_iter_percent_idx += 1
                    curr_iter += 1
                    if curr_iter_percent_idx == len(percent_idx):
                        break
                if curr_iter_percent_idx == len(percent_idx):
                    break
        assert (done == len(percent_idx))
        batch_error_dict = evaluation.compute_batch_statistics(predictions_new,
                                                               scene.dt,
                                                               max_hl=max_hl,
                                                               ph=ph,
                                                               node_type_enum=env.NodeType,
                                                               map=None,
                                                               best_of=True,
                                                               prune_ph_to_future=True)
        eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
        eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
        eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))
        return eval_ade_batch_errors, eval_fde_batch_errors, eval_kde_nll


if __name__ == "__main__":
    nb_classes = 3
    if args.ldam == "yes":
        from model.trajectron_multi_ldam import Trajectron
        joint_train = True
    else:
        from model.trajectron_multi import Trajectron
        joint_train = False
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams, borders = load_model(args.model, env, args.ldam, ts=args.checkpoint)
    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes
    env_name = scenes[0].name
    with open(os.path.join(args.scores_dir, '%s_kalman.pkl' % env_name), 'rb') as f:
        scores = dill.load(f)
    scores_sorted = np.sort(scores)
    hardest_1_percent = scores_sorted[-int(len(scores_sorted) / 100)]
    hardest_3_percent = scores_sorted[-int(len(scores_sorted) / 100) * 3]
    hardest_5_percent = scores_sorted[-int(len(scores_sorted) / 100) * 5]

    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']
    with torch.no_grad():
        ############### BEST OF 20 ###############
        eval_ade_batch_errors = np.array([])
        eval_fde_batch_errors = np.array([])
        eval_kde_nll = np.array([])
        eval_ade_batch_errors_1_percent = np.array([])
        eval_fde_batch_errors_1_percent = np.array([])
        eval_kde_nll_1_percent = np.array([])
        eval_ade_batch_errors_3_percent = np.array([])
        eval_fde_batch_errors_3_percent = np.array([])
        eval_kde_nll_3_percent = np.array([])
        eval_ade_batch_errors_5_percent = np.array([])
        eval_fde_batch_errors_5_percent = np.array([])
        eval_kde_nll_5_percent = np.array([])
        eval_statistics_details = {}
        for i in range(nb_classes):
            eval_statistics_details[i] = {}
            eval_statistics_details[i]['eval_ade_batch_errors'] = np.array([])
            eval_statistics_details[i]['eval_fde_batch_errors'] = np.array([])
            eval_statistics_details[i]['eval_kde_nll'] = np.array([])
        epes = []
        predictions_cl_list = []
        current_epes = []
        current_epe_list = []
        predictions_cl_list = []
        print("-- Evaluating best of 20")
        for i, scene in enumerate(scenes):
            print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
            for t in tqdm(range(0, scene.timesteps, 10)):
                timesteps = np.arange(t, t + 10)
                if args.ldam == "yes":
                    predictions, features, predictions_cl = eval_stg.predict(scene,
                                                                             timesteps,
                                                                             ph,
                                                                             joint_train=joint_train,
                                                                             num_samples=20,
                                                                             min_history_timesteps=7,
                                                                             min_future_timesteps=12,
                                                                             z_mode=False,
                                                                             gmm_mode=False,
                                                                             full_dist=False)

                    if not predictions:
                        continue
                    predictions_cl_list.append(predictions_cl)
                else:
                    predictions, features = eval_stg.predict(scene,
                                                             timesteps,
                                                             ph,
                                                             num_samples=20,
                                                             min_history_timesteps=7,
                                                             min_future_timesteps=12,
                                                             z_mode=False,
                                                             gmm_mode=False,
                                                             full_dist=False)

                    if not predictions:
                        continue
                (prediction_dict, histories_dict, futures_dict) = prediction_output_to_trajectories(predictions,
                                                                                                    scene.dt,
                                                                                                    max_hl,
                                                                                                    ph,
                                                                                                    prune_ph_to_future=True)

                for t in prediction_dict.keys():
                    for node in prediction_dict[t].keys():
                        z_future = get_kalman_filter_result(histories_dict[t][node])
                        epe = calculate_epe(z_future, futures_dict[t][node][-1, :])
                        #print('epe ', epe, ' class ', get_class_label(epe, borders, nb_classes))
                        current_epe_list.append(epe)
                        epes.append(get_class_label(epe, borders, nb_classes))
                        current_epes.append(get_class_label(epe, borders, nb_classes))

                one_percent_idx = np.where(np.array(current_epe_list) >= hardest_1_percent)[0]
                three_percent_idx = np.where(np.array(current_epe_list) >= hardest_3_percent)[0]
                five_percent_idx = np.where(np.array(current_epe_list) >= hardest_5_percent)[0]

                batch_error_dict, batch_error_dict_details = evaluation.compute_batch_statistics_classes(predictions,
                                                                                                         scene.dt,
                                                                                                         current_epes,
                                                                                                         nb_classes,
                                                                                                         max_hl=max_hl,
                                                                                                         ph=ph,
                                                                                                         node_type_enum=env.NodeType,
                                                                                                         map=None,
                                                                                                         best_of=True,
                                                                                                         prune_ph_to_future=True)

                eval_ade_batch_errors_1_percent, eval_fde_batch_errors_1_percent, eval_kde_nll_1_percent = get_eval_percent(
                    eval_ade_batch_errors_1_percent, eval_fde_batch_errors_1_percent, eval_kde_nll_1_percent, one_percent_idx, current_epe_list, predictions)
                eval_ade_batch_errors_3_percent, eval_fde_batch_errors_3_percent, eval_kde_nll_3_percent = get_eval_percent(
                    eval_ade_batch_errors_3_percent, eval_fde_batch_errors_3_percent, eval_kde_nll_3_percent, three_percent_idx, current_epe_list, predictions)
                eval_ade_batch_errors_5_percent, eval_fde_batch_errors_5_percent, eval_kde_nll_5_percent = get_eval_percent(
                    eval_ade_batch_errors_5_percent, eval_fde_batch_errors_5_percent, eval_kde_nll_5_percent, five_percent_idx, current_epe_list, predictions)

                eval_ade_batch_errors = np.hstack((eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                eval_fde_batch_errors = np.hstack((eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))
                for i in range(nb_classes):
                    if len(batch_error_dict_details[args.node_type][i]['ade']) > 0:
                        eval_statistics_details[i]['eval_ade_batch_errors'] = np.hstack((eval_statistics_details[i]['eval_ade_batch_errors'], batch_error_dict_details[args.node_type][i]['ade']))
                        eval_statistics_details[i]['eval_fde_batch_errors'] = np.hstack((eval_statistics_details[i]['eval_fde_batch_errors'], batch_error_dict_details[args.node_type][i]['fde']))
                        eval_statistics_details[i]['eval_kde_nll'] = np.hstack((eval_statistics_details[i]['eval_kde_nll'], batch_error_dict_details[args.node_type][i]['kde']))
                current_epes = []
                current_epe_list = []
        epes = np.array(epes)
        for i in range(nb_classes):
            pd.DataFrame({'value': eval_statistics_details[i]['eval_ade_batch_errors'], 'metric': 'ade', 'type': 'best_of'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_class_' + str(i) + '_ade_best_of.csv'))
            pd.DataFrame({'value': eval_statistics_details[i]['eval_fde_batch_errors'], 'metric': 'fde', 'type': 'best_of'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_class_' + str(i) + '_fde_best_of.csv'))
            pd.DataFrame({'value': eval_statistics_details[i]['eval_kde_nll'], 'metric': 'kde', 'type': 'best_of'}
                         ).to_csv(os.path.join(args.output_path, args.output_tag + '_class_' + str(i) + '_kde_best_of.csv'))

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_ade_best_of.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_fde_best_of.csv'))
        pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_kde_best_of.csv'))

        pd.DataFrame({'value': eval_ade_batch_errors_1_percent, 'metric': 'ade', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_1_percent' + '_ade_best_of.csv'))
        pd.DataFrame({'value': eval_ade_batch_errors_1_percent, 'metric': 'fde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_1_percent' + '_fde_best_of.csv'))
        pd.DataFrame({'value': eval_ade_batch_errors_1_percent, 'metric': 'kde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_1_percent' + '_kde_best_of.csv'))

        pd.DataFrame({'value': eval_ade_batch_errors_3_percent, 'metric': 'ade', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_3_percent' + '_ade_best_of.csv'))
        pd.DataFrame({'value': eval_ade_batch_errors_3_percent, 'metric': 'fde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_3_percent' + '_fde_best_of.csv'))
        pd.DataFrame({'value': eval_ade_batch_errors_3_percent, 'metric': 'kde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_3_percent' + '_kde_best_of.csv'))

        pd.DataFrame({'value': eval_ade_batch_errors_5_percent, 'metric': 'ade', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_5_percent' + '_ade_best_of.csv'))
        pd.DataFrame({'value': eval_ade_batch_errors_5_percent, 'metric': 'fde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_5_percent' + '_fde_best_of.csv'))
        pd.DataFrame({'value': eval_ade_batch_errors_5_percent, 'metric': 'kde', 'type': 'best_of'}
                     ).to_csv(os.path.join(args.output_path, args.output_tag + '_hardest_5_percent' + '_kde_best_of.csv'))

        if args.ldam == 'yes':
            predictions_cl_list = np.hstack(predictions_cl_list)
            print('accuracy : ', sum(1 for x, y in zip(epes, predictions_cl_list) if x == y) / float(len(epes)))
            for i in range(nb_classes):
                epe_idx_class = [j for j, element in enumerate(epes) if element == i]
                epe_class = epes[epe_idx_class]
                print('accuracy of class ', i, ' : ', sum(1 for x, y in zip(epe_class, predictions_cl_list[epe_idx_class]) if x == y) / float(len(epe_class)))
