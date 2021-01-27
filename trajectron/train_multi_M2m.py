import json
import os
import pathlib
import pdb
import random
import time
import warnings

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim, utils
from tqdm import tqdm

import evaluation
import visualization
from argument_parser import args
from model.dataset import EnvironmentDataset, EnvironmentDatasetKalman, collate
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.trajectron_M2m import Trajectron


def train_epoch(trajectron, train_data_loader, epoch):
    """
    """
    global curr_iter_node_type, lr_scheduler, optimizer, log_writer
    loss_epoch = []
    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        pbar = tqdm(data_loader, ncols=80)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            optimizer[node_type].zero_grad()
            train_loss = trajectron.train_loss(batch, node_type)
            pbar.set_description(
                f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f}")
            loss_epoch.append(train_loss.item())
            train_loss.backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(
                    trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
            optimizer[node_type].step()
            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler[node_type].step()
            curr_iter += 1

        if not args.debug:
            log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                  lr_scheduler[node_type].get_lr()[0],
                                  epoch)
            log_writer.add_scalar(
                f"{node_type}/train/loss", np.mean(loss_epoch), epoch)
        curr_iter_node_type[node_type] = curr_iter
    return np.mean(loss_epoch)


if __name__ == '__main__':

    # torch.autograd.set_detect_anomaly(True)
    if not torch.cuda.is_available() or args.device == 'cpu':
        args.device = torch.device('cpu')
    else:
        if torch.cuda.device_count() == 1:
            # If you have CUDA_VISIBLE_DEVICES set, which you should,
            # then this will prevent leftover flag arguments from
            # messing with the device allocation.
            args.device = 'cuda:0'

        args.device = torch.device(args.device)

    if args.eval_device is None:
        args.eval_device = torch.device('cpu')

    # This is needed for memory pinning using a DataLoader (otherwise memory is pinned to cuda:0 by default)
    torch.cuda.set_device(args.device)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['dynamic_edges'] = args.dynamic_edges
    hyperparams['edge_state_combine_method'] = args.edge_state_combine_method
    hyperparams['edge_influence_combine_method'] = args.edge_influence_combine_method
    hyperparams['edge_addition_filter'] = args.edge_addition_filter
    hyperparams['edge_removal_filter'] = args.edge_removal_filter
    hyperparams['batch_size'] = args.batch_size
    hyperparams['k_eval'] = args.k_eval
    hyperparams['offline_scene_graph'] = args.offline_scene_graph
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    hyperparams['use_map_encoding'] = args.map_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius
    hyperparams['Enc_FC_dims'] = 128
    hyperparams['num_hyp'] = 20

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Offline Scene Graph Calculation: %s' % args.offline_scene_graph)
    print('| EE state_combine_method: %s' % args.edge_state_combine_method)
    print('| EIE scheme: %s' % args.edge_influence_combine_method)
    print('| dynamic_edges: %s' % args.dynamic_edges)
    print('| robot node: %s' % args.incl_robot_node)
    print('| edge_addition_filter: %s' % args.edge_addition_filter)
    print('| edge_removal_filter: %s' % args.edge_removal_filter)
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 'models_Multi_hyp' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    # ! Load training and evaluation environments and scenes
    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    scores_path = args.scores_dir
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(
            ' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(
            attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        # TODO: Make more general, allow the user to specify?
        train_env.robot_type = train_env.NodeType[0]
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    train_dataset = EnvironmentDatasetKalman(train_env,
                                             scores_path,
                                             hyperparams['state'],
                                             hyperparams['pred_state'],
                                             scene_freq_mult=hyperparams['scene_freq_mult_train'],
                                             node_freq_mult=hyperparams['node_freq_mult_train'],
                                             hyperparams=hyperparams,
                                             min_history_timesteps=hyperparams['minimum_history_length'],
                                             min_future_timesteps=hyperparams['prediction_horizon'],
                                             return_robot=not args.incl_robot_node)
    train_data_loader = dict()

    for node_type_data_set in train_dataset:
        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     shuffle=True,
                                                     num_workers=args.preprocess_workers)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")
    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        eval_data_path = os.path.join(args.data_dir, args.eval_data_dict)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                ' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(
                attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            # TODO: Make more general, allow the user to specify?
            eval_env.robot_type = eval_env.NodeType[0]
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDataset(eval_env,
                                          hyperparams['state'],
                                          hyperparams['pred_state'],
                                          scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                          node_freq_mult=hyperparams['node_freq_mult_eval'],
                                          hyperparams=hyperparams,
                                          min_history_timesteps=hyperparams['minimum_history_length'],
                                          min_future_timesteps=hyperparams['prediction_horizon'],
                                          return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if args.eval_device is 'cpu' else True,
                                                         batch_size=args.eval_batch_size,
                                                         shuffle=True,
                                                         num_workers=args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_data_path}")
    # ! Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")
    # ! Creating Models
    if args.net_g_dir and args.net_g_ts:
        model_registrar_g = ModelRegistrar(args.net_g_dir, args.device)
        model_registrar_g.load_models(args.net_g_ts)
        trajectron_g = Trajectron(model_registrar_g,
                                  hyperparams,
                                  log_writer,
                                  args.device)
        trajectron_g.set_environment(train_env)
    else:
        model_dir = model_dir + "_g"

    model_registrar = ModelRegistrar(model_dir, args.device)
    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)
    trajectron.set_environment(train_env)

    print('Created Training Model.')

    # eval_trajectron = None
    # if args.eval_every is not None or args.vis_every is not None:
    #     eval_trajectron = Trajectron(model_registrar,
    #                                  hyperparams,
    #                                  log_writer,
    #                                  args.eval_device)
    #     eval_trajectron.set_environment(eval_env)
    #     # eval_trajectron.set_annealing_params()
    # print('Created Evaluation Model.')

    # ! Defining optimizers
    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr': 0.0008}], lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])
    # ! Training loop
    train_loss_df = pd.DataFrame(columns=['epoch', 'loss'])
    eval_loss_df = pd.DataFrame(columns=['epoch', 'loss'])
    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {
        node_type: 0 for node_type in train_data_loader.keys()}
    for epoch in range(1, args.train_epochs + 1):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        if epoch >= args.warm and args.gen:
            # Generation process and training with generated data
            pass
        else:
            loss_epoch = train_epoch(trajectron, train_data_loader, epoch)
            train_loss_df = train_loss_df.append(pd.DataFrame(data=[[epoch, loss_epoch], columns=[
                'epoch', 'loss']), ignore_index=True)
        train_dataset.augment = False

        # if args.eval_every is not None or args.vis_every is not None:
        #     eval_trajectron.set_curr_iter(epoch)

        #################################
        #           EVALUATION          #
        #################################
        # if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
        #     max_hl = hyperparams['maximum_history_length']
        #     ph = hyperparams['prediction_horizon']
        #     model_registrar.to(args.eval_device)
        #     with torch.no_grad():
        #         # Calculate evaluation loss
        #         for node_type, data_loader in eval_data_loader.items():
        #             eval_loss = []
        #             print(
        #                 f"Starting Multi Hyp Evaluation @ epoch {epoch} for node type: {node_type}")
        #             pbar = tqdm(data_loader, ncols=80)
        #             loss_epoch = []
        #             for batch in pbar:
        #                 eval_loss_node_type = eval_trajectron.eval_loss(
        #                     batch, node_type)
        #                 pbar.set_description(
        #                     f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
        #                 loss_epoch.append(eval_loss_node_type.item())

        #                 eval_loss.append(
        #                     {node_type: {'wta': [eval_loss_node_type]}})
        #                 del batch
        #             eval_loss_df = eval_loss_df.append(pd.DataFrame(
        #                 [[epoch, np.mean(loss_epoch)]], columns=['epoch', 'loss']), ignore_index=True)

        #             evaluation.log_batch_errors(eval_loss,
        #                                         log_writer,
        #                                         f"{node_type}/eval_loss",
        #                                         epoch)

        # if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
        #     model_registrar.save_models(epoch)
    train_loss_df.to_csv('./training_logs/train_loss%s.csv' % args.log_tag, sep=";")
    eval_loss_df.to_csv('./training_logs/eval_loss%s.csv' % args.log_tag, sep=";")
