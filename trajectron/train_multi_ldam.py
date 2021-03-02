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
from model.trajectron_multi_ldam import Trajectron

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

def get_losses_accuracies(train_loss, train_loss_no_reweighted, train_logits, batch_targets, nb_classes):
    i = 0
    losses = {}
    losses_n_w = {}
    accuracies = {}
    for i in range(nb_classes):
        class_idx = (batch_targets== i).nonzero()
        if len(class_idx) == 0:
            pass
        else:
            train_logits_class = train_logits[class_idx]
            train_loss_class = round(train_loss[class_idx].mean().item(), 2)
            train_loss_class_n_w = round(train_loss_no_reweighted[class_idx].mean().item(), 2)
            losses[i] = train_loss_class
            losses_n_w[i] = train_loss_class_n_w
            accuracies[i] = round(((train_logits_class==i).sum()).item()/ len(class_idx), 2)*100
    accuracy = round ((train_logits.cpu() == batch_targets).sum().item()/ len(batch_targets), 2)*100
    
    return losses, losses_n_w, accuracies, accuracy

def epoch_losses_accuracies(losses_list, losses_list_n_w, accuracies_list, nb_classes, epoch):
    epoch_loss = {}
    epoch_loss_n_w = {}
    epoch_accuracies = {}
    for i in range(nb_classes):
        epoch_loss['epoch'] = epoch
        epoch_loss_n_w['epoch'] = epoch
        epoch_accuracies['epoch'] = epoch
        loss = []
        loss_n_w = []
        accuracy = []
        for j in range(len(losses_list)):
            if i in losses_list[j]:
                loss.append(losses_list[j][i])
                loss_n_w.append(losses_list_n_w[j][i])
                accuracy.append(accuracies_list[j][i])
        if len(loss) > 0:
            epoch_loss[i] = round(np.mean(loss), 2)
            epoch_loss_n_w[i] = round(np.mean(loss_n_w), 2)
            epoch_accuracies[i] = round(np.mean(accuracy), 2)
    return epoch_loss, epoch_loss_n_w, epoch_accuracies

def main():
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
    
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir,
                                 'models_Multi_hyp' + time.strftime('%d_%b_%Y_%H_%M_%S', time.localtime()) + args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    load_model = True
    
    if load_model:
        print('######### LOADING MODEL #####################')
        #model_dir_loaded = '/misc/lmbraid21/ayadim/Trajectron-plus-plus/experiments/pedestrians/models/models_Multi_hyp16_Jan_2021_19_56_41_hotel_ar3'
        #model_dir_loaded = '/misc/lmbraid21/ayadim/Trajectron-plus-plus/experiments/pedestrians/models/models_Multi_hyp13_Jan_2021_23_20_13_eth_ar3/model_classification_22_02_2021-18_50_cosann_10_2_ce_2_conloss_eth_ar3'
        model_dir_loaded = '/misc/lmbraid21/ayadim/Trajectron-plus-plus/experiments/pedestrians/models/models_Multi_hyp02_Mar_2021_eth_ar3_contrastive_lr001_3classes_'
        ts = 150
    else:
        print('######### Training from scratch #####################')
        ts = 150
    # Load training and evaluation environments and scenes
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
                                             binary = False, 
                                             min_history_timesteps=hyperparams['minimum_history_length'],
                                             min_future_timesteps=hyperparams['prediction_horizon'],
                                             return_robot=not args.incl_robot_node)
    # nb of observations in each Kalman class
    class_count_dict = train_dataset.class_count_dict 
    nb_classes = list(class_count_dict[0].keys())[-1] +1
    print('classes: ', class_count_dict)
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
    '''eval_scenes = []
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
'''
    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")
        '''
        for i, scene in enumerate(eval_scenes):
            scene.calculate_scene_graph(eval_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Evaluation Scene {i}")
            '''

    if load_model:
        model_registrar = ModelRegistrar(model_dir_loaded, args.device)
        model_registrar.load_models(ts)
    else:
        model_registrar = ModelRegistrar(model_dir, args.device)

    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device,
                            class_count_dict)
    trajectron.set_environment(train_env)
    # trajectron.set_annealing_params()
    print('Created Training Model.')
    '''
    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env)
        # eval_trajectron.set_annealing_params()
    print('Created Evaluation Model.')
    '''
    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr': 0.0008}], 
                                           lr=hyperparams['learning_rate'])#, weight_decay = 2e-4)
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            print('######################### const lr ######################')
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            print('######################### exp lr scheduler ######################')
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type],
                                                                       gamma=hyperparams['learning_decay_rate'])
        elif hyperparams['learning_rate_style'] == 'cosAnnWarmRestart':
            print('######################### cosAnnWarmRestart lr scheduler ######################')
            lr_scheduler[node_type] = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer[node_type],
                                                                    eta_min = 0,
                                                                    T_0 = hyperparams['T_0'],
                                                                    T_mult = hyperparams['T_mult'],)

    print('optimizer: ', optimizer)
    train_loss_df = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])
    eval_loss_df = pd.DataFrame(columns=['epoch', 'loss', 'accuracy'])
    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {
        node_type: 0 for node_type in train_data_loader.keys()}
    top_n = hyperparams['num_hyp']
    weight = None
    train_accuracies_list = []
    train_losses_list = []
    iters = 0
    for node_type, data_loader in train_data_loader.items():
        # used if optimizer cos annealing warm restarts
        iters += len(data_loader)

    for epoch in range(1 + ts, args.train_epochs + 1 + ts):
        print ('Epoch ', epoch, ' lr: ', lr_scheduler[node_type].get_lr()[0])
        if epoch > ts + 100 :
            weight = torch.cuda.FloatTensor([*class_count_dict[0].values()])
            weight = 1.0 / weight
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        loss_epoch = []
        loss_n_w_epoch = []
        accuracy_epoch = []
        losses_list = []
        losses_n_w_list = []
        accuracies_list = []

        for node_type, data_loader in train_data_loader.items():
            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80)
            for batch in pbar:
                trajectron.set_curr_iter(curr_iter)
                # trajectron.step_annealers(node_type)
                optimizer[node_type].zero_grad()
                batch_targets= batch[-1]
                train_loss, train_loss_no_reweighted, train_logits = trajectron.train_loss(
                    batch, node_type, weight = weight, top_n=top_n)
                
                losses, losses_n_w, accuracies, accuracy = get_losses_accuracies(train_loss,train_loss_no_reweighted,  train_logits, batch_targets, nb_classes)
                losses_list.append(losses)
                accuracies_list.append(accuracies)
                losses_n_w_list.append(losses_n_w)

                pbar.set_description(
                    f"Epoch {epoch}, {node_type} L: {train_loss.mean().item():.2f} L_n_w: {train_loss_no_reweighted.mean().item():.2f} A: {accuracy:.2f}")
                
                
                loss_epoch.append(train_loss.mean().item())
                loss_n_w_epoch.append(train_loss_no_reweighted.mean().item())
                accuracy_epoch.append(accuracy)
                train_loss.mean().backward()
                # Clipping gradients.
                if hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(
                        model_registrar.parameters(), hyperparams['grad_clip'])
                optimizer[node_type].step()

                curr_iter += 1
            
            # Stepping forward the learning rate scheduler and annealers.
            if hyperparams['learning_rate_style'] == 'cosAnnWarmRestart':
                lr_scheduler[node_type].step()
            else:
                lr_scheduler[node_type].step()

                
            epoch_losses, epoch_losses_n_w, epoch_accuracies = epoch_losses_accuracies(losses_list, losses_n_w_list, accuracies_list, nb_classes, epoch)
            print('Epoch ', epoch, ' losses: ', epoch_losses)
            print('Epoch ', epoch, ' losses non weighted: ', epoch_losses_n_w)
            print('Epoch ', epoch, ' accuracies: ', epoch_accuracies)
            if not args.debug:
                log_writer.add_scalar(f"{node_type}/train/learning_rate",
                                      lr_scheduler[node_type].get_lr()[0],
                                      epoch)
                log_writer.add_scalar(
                    f"{node_type}/train/loss", np.mean(loss_epoch), epoch)
                log_writer.add_scalar(
                    f"{node_type}/train/loss_non_weighted", np.mean(loss_n_w_epoch), epoch)
                log_writer.add_scalar(
                    f"{node_type}/train/accuracy", np.mean(accuracy_epoch), epoch)
                for i in range(nb_classes):
                    log_writer.add_scalar(
                        f"{node_type}/train/loss_class_"+str(i), epoch_losses[i], epoch)
                    log_writer.add_scalar(
                        f"{node_type}/train/loss_n_w_class_"+str(i), epoch_losses_n_w[i], epoch)
                    #log_writer.add_scalar(
                    #    f"{node_type}/train/loss_minority", epoch_losses[nb_classes-1], epoch)
                    log_writer.add_scalar(
                        f"{node_type}/train/accuracy_class_"+str(i), epoch_accuracies[i], epoch)

            curr_iter_node_type[node_type] = curr_iter
            train_loss_df = train_loss_df.append(pd.DataFrame(data=[[epoch, np.mean(loss_epoch), np.mean(accuracy_epoch)]], columns=[
                'epoch', 'loss', 'accuracy']), ignore_index=True)
            train_losses_list.append(epoch_losses)
            train_accuracies_list.append(epoch_accuracies)
        train_dataset.augment = False

        '''if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        # if args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0:
        #     max_hl = hyperparams['maximum_history_length']
        #     ph = hyperparams['prediction_horizon']
        #     with torch.no_grad():
        #         # Predict random timestep to plot for train data set
        #         if args.scene_freq_mult_viz:
        #             scene = np.random.choice(
        #                 train_scenes, p=train_scenes_sample_probs)
        #         else:
        #             scene = np.random.choice(train_scenes)
        #         timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
        #         predictions = trajectron.predict(scene,
        #                                          timestep,
        #                                          ph,
        #                                          z_mode=True,
        #                                          gmm_mode=True,
        #                                          all_z_sep=False,
        #                                          full_dist=False)

        #         # Plot predicted timestep for random scene
        #         fig, ax = plt.subplots(figsize=(10, 10))
        #         visualization.visualize_prediction(ax,
        #                                            predictions,
        #                                            scene.dt,
        #                                            max_hl=max_hl,
        #                                            ph=ph,
        #                                            map=scene.map['VISUALIZATION'] if scene.map is not None else None)
        #         ax.set_title(f"{scene.name}-t: {timestep}")
        #         log_writer.add_figure('train/prediction', fig, epoch)

        #         model_registrar.to(args.eval_device)
        #         # Predict random timestep to plot for eval data set
        #         if args.scene_freq_mult_viz:
        #             scene = np.random.choice(
        #                 eval_scenes, p=eval_scenes_sample_probs)
        #         else:
        #             scene = np.random.choice(eval_scenes)
        #         timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
        #         predictions = eval_trajectron.predict(scene,
        #                                               timestep,
        #                                               ph,
        #                                               num_samples=20,
        #                                               min_future_timesteps=ph,
        #                                               z_mode=False,
        #                                               full_dist=False)

        #         # Plot predicted timestep for random scene
        #         fig, ax = plt.subplots(figsize=(10, 10))
        #         visualization.visualize_prediction(ax,
        #                                            predictions,
        #                                            scene.dt,
        #                                            max_hl=max_hl,
        #                                            ph=ph,
        #                                            map=scene.map['VISUALIZATION'] if scene.map is not None else None)
        #         ax.set_title(f"{scene.name}-t: {timestep}")
        #         log_writer.add_figure('eval/prediction', fig, epoch)

        #         # Predict random timestep to plot for eval data set
        #         predictions = eval_trajectron.predict(scene,
        #                                               timestep,
        #                                               ph,
        #                                               min_future_timesteps=ph,
        #                                               z_mode=True,
        #                                               gmm_mode=True,
        #                                               all_z_sep=True,
        #                                               full_dist=False)

        #         # Plot predicted timestep for random scene
        #         fig, ax = plt.subplots(figsize=(10, 10))
        #         visualization.visualize_prediction(ax,
        #                                            predictions,
        #                                            scene.dt,
        #                                            max_hl=max_hl,
        #                                            ph=ph,
        #                                            map=scene.map['VISUALIZATION'] if scene.map is not None else None)
        #         ax.set_title(f"{scene.name}-t: {timestep}")
        #         log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    eval_loss = []
                    print(
                        f"Starting Multi Hyp Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    loss_epoch = []
                    for batch in pbar:
                        eval_loss_node_type = eval_trajectron.eval_loss(
                            batch, node_type)
                        pbar.set_description(
                            f"Epoch {epoch}, {node_type} L: {eval_loss_node_type.item():.2f}")
                        loss_epoch.append(eval_loss_node_type.item())

                        eval_loss.append(
                            {node_type: {'wta': [eval_loss_node_type]}})
                        del batch
                    eval_loss_df = eval_loss_df.append(pd.DataFrame(
                        [[epoch, np.mean(loss_epoch)]], columns=['epoch', 'loss']), ignore_index=True)

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)
                '''
                # Predict batch timesteps for evaluation dataset evaluation
                # eval_batch_errors = []
                # for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                #     timesteps = scene.sample_timesteps(args.eval_batch_size)

                #     predictions = eval_trajectron.predict(scene,
                #                                           timesteps,
                #                                           ph,
                #                                           num_samples=50,
                #                                           min_future_timesteps=ph,
                #                                           full_dist=False)

                #     eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                #                                                                  scene.dt,
                #                                                                  max_hl=max_hl,
                #                                                                  ph=ph,
                #                                                                  node_type_enum=eval_env.NodeType,
                #                                                                  map=scene.map))

                # evaluation.log_batch_errors(eval_batch_errors,
                #                             log_writer,
                #                             'eval',
                #                             epoch,
                #                             bar_plot=['kde'],
                #                             box_plot=['ade', 'fde'])

                # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                # eval_batch_errors_ml = []
                # for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                #     timesteps = scene.sample_timesteps(scene.timesteps)

                #     predictions = eval_trajectron.predict(scene,
                #                                           timesteps,
                #                                           ph,
                #                                           num_samples=1,
                #                                           min_future_timesteps=ph,
                #                                           z_mode=True,
                #                                           gmm_mode=True,
                #                                           full_dist=False)

                #     eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                #                                                                     scene.dt,
                #                                                                     max_hl=max_hl,
                #                                                                     ph=ph,
                #                                                                     map=scene.map,
                #                                                                     node_type_enum=eval_env.NodeType,
                #                                                                     kde=False))

                # evaluation.log_batch_errors(eval_batch_errors_ml,
                #                             log_writer,
                #                             'eval/ml',
                #                             epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch, args.log_tag)
    train_loss_df.to_csv(os.path.join(model_dir, 'train_loss%s.csv' % args.log_tag), sep=";")
    losses_file = os.path.join(model_dir, 'losses'+ args.log_tag +'.json') 
    with open(losses_file, 'w') as fout:
        json.dump(train_losses_list, fout)
    accuracies_file = os.path.join(model_dir, 'accuracies'+ args.log_tag +'.json') 
    with open(accuracies_file, 'w') as fout:
        json.dump(train_accuracies_list, fout)
    #eval_loss_df.to_csv('eval_loss%s.csv' % args.log_tag, sep=";")


if __name__ == '__main__':
    main()
