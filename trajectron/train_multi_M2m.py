import json
import os
import pathlib
import random
import time
from datetime import datetime
import warnings

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch import log, nn, optim, utils
from torch.utils.data.sampler import Sampler
from tqdm import tqdm
from shutil import copyfile

import evaluation
import visualization
from argument_parser import args
# from m2m_toolbox import FocalLoss, bcolors, generation, train_epoch, train_epoch_con_score_based, train_gen_epoch, train_epoch_con, train_epoch_con_score_based, LDAMLoss, SupervisedConLoss, ScoreBasedConLoss
from m2m_toolbox import *
from model.dataset import EnvironmentDataset, EnvironmentDatasetKalman, collate
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.trajectron_M2m import Trajectron

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

    print("\n#################################")
    print("#       M2m-Trajectron++        #")
    print("#################################")

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
    if args.model_tag:
        model_tag = args.model_tag
    else:
        model_tag = "_".join(["model_classification", datetime.now().strftime("%d_%m_%Y-%H_%M"), args.log_tag])
    # model_tag = "model_classification_22_02_2021-12_27_cosann_10_2_ce_2_conloss_eth_ar3"
    if args.gen:
        # if we are generating (create subfolder for f)
        model_dir_f = os.path.join(args.log_dir, args.experiment, model_tag, model_tag + '_f_edge_and_hist_' + datetime.now().strftime("%d_%m_%Y-%H_%M"))
        pathlib.Path(model_dir_f).mkdir(parents=True, exist_ok=True)
        model_dir_g = os.path.join(args.log_dir, args.experiment, model_tag)
        checkpoint_name = 'model_registrar-%d-%s.pt' % (args.net_g_ts, args.net_g_extra_tag)
        copyfile(os.path.join(model_dir_g, checkpoint_name), os.path.join(model_dir_f, checkpoint_name))
    if not args.debug:
        # Create the log and model directiory if they're not present.
        if not args.gen:
            model_dir = os.path.join(args.log_dir, args.experiment, model_tag)
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)
        else:
            model_dir = model_dir_f
        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    print(model_dir)
    # Load training and evaluation environments and scenes
    train_scenes = []
    train_data_path = os.path.join(args.data_dir, args.train_data_dict)
    scores_path = args.scores_dir
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

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

    hyperparams['class_count_dic'] = train_dataset.class_count_dict[0]
    hyperparams['class_count'] = list(hyperparams['class_count_dic'].values())
    hyperparams['class_weights'] = train_dataset.class_weights[0]
    hyperparams['num_classes'] = len(hyperparams['class_count_dic'])
    # TODO Read these values from command line args
    hyperparams['beta'] = 0.9  # (0.9, 0.99, 0.999) Lower -> bigger p accept
    # lower acceptance bound on logit for g: L(g;x*,k)
    hyperparams['gamma'] = 0.8  # (0.9, 0.99) Lower -> bigger p accept
    hyperparams['lam'] = 0.1  # (0.01, 0.1, 0.5) Lower -> bigger p accept
    hyperparams['step_size'] = 0.1
    hyperparams['attack_iter'] = 10
    hyperparams['non_linearity'] = 'none'
    hyperparams['data_loader_sampler'] = 'random'
    # hyperparams['learning_rate_style'] = 'cosannw'
    # hyperparams['learning_rate'] = 0.01  # Override lr

    N_SAMPLES_PER_CLASS_T = torch.Tensor(hyperparams['class_count']).to(args.device)

    train_data_loader = dict()

    for node_type_data_set in train_dataset:
        log_writer.add_histogram(tag=f"{node_type_data_set.node_type}Kalman scores histogram",
                                 values=node_type_data_set.scores)
        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                     collate_fn=collate,
                                                     pin_memory=False if args.device is 'cpu' else True,
                                                     batch_size=args.batch_size,
                                                     #  sampler=node_type_data_set.weighted_sampler,
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
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            # TODO: Make more general, allow the user to specify?
            eval_env.robot_type = eval_env.NodeType[0]
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDatasetKalman(eval_env,
                                                scores_path,
                                                hyperparams['state'],
                                                hyperparams['pred_state'],
                                                scene_freq_mult=hyperparams['scene_freq_mult_eval'],
                                                node_freq_mult=hyperparams['node_freq_mult_eval'],
                                                hyperparams=hyperparams,
                                                min_history_timesteps=hyperparams['minimum_history_length'],
                                                min_future_timesteps=hyperparams['prediction_horizon'],
                                                return_robot=not args.incl_robot_node,
                                                borders=train_dataset.boarders[0])

        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                         collate_fn=collate,
                                                         pin_memory=False if args.eval_device is 'cpu' else True,
                                                         batch_size=args.eval_batch_size,
                                                         shuffle=True,
                                                         num_workers=args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader
    # TODO Make sure that the number of classes are the same for training, eval, test
        print(f"Loaded evaluation data from {eval_data_path}")
    # Offline Calculate Scene Graph
    if hyperparams['offline_scene_graph'] == 'yes':
        print(f"Offline calculating scene graphs")
        print("Training scene graphs")
        for i, scene in enumerate(train_scenes):
            scene.calculate_scene_graph(train_env.attention_radius,
                                        hyperparams['edge_addition_filter'],
                                        hyperparams['edge_removal_filter'])
            print(f"Created Scene Graph for Training Scene {i}")

        if args.eval_every is not None or args.vis_every is not None:
            print("Evaluation scene graphs")
            for i, scene in enumerate(eval_scenes):
                scene.calculate_scene_graph(eval_env.attention_radius,
                                            hyperparams['edge_addition_filter'],
                                            hyperparams['edge_removal_filter'])
                print(f"Created Scene Graph for Evaluation Scene {i}")
    # Creating Models
    if args.net_g_ts:
        print("Loading baseline classifier g:")
        model_registrar_g = ModelRegistrar(model_dir, args.device)
        model_registrar_g.load_models(args.net_g_ts, args.net_g_extra_tag)
        trajectron_g = Trajectron(model_registrar_g,
                                  hyperparams,
                                  log_writer,
                                  args.device)
        trajectron_g.set_environment(train_env)
        extra_tag = "f_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s" % (hyperparams['beta'], hyperparams['gamma'], hyperparams['lam'],
                                                         hyperparams['step_size'], hyperparams['attack_iter'],
                                                         hyperparams['learning_rate'], str(args.batch_size),
                                                         hyperparams['non_linearity'], hyperparams['data_loader_sampler'],
                                                         hyperparams['num_classes'])
    else:
        extra_tag = "g_%s_%s_%s_%s_%s" % (hyperparams['learning_rate'], str(args.batch_size),
                                          hyperparams['non_linearity'], hyperparams['data_loader_sampler'],
                                          hyperparams['num_classes'])
    # Load pretrained model on multi hypothesis
    model_registrar = ModelRegistrar(model_dir, args.device)

    if args.net_trajectron_ts:
        import pdb; pdb.set_trace()
        model_registrar.load_models(args.net_trajectron_ts, extra_tag=extra_tag)
    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)
    trajectron.set_environment(train_env)

    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env)
    print('Created Evaluation Model.')

    # Defining optimizers
    optimizer = dict()
    lr_scheduler = dict()
    initial_lr_state = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.Adam([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()},
                                           {'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr': 0.0008}],
                                          lr=hyperparams['learning_rate'])
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=hyperparams['learning_decay_rate'])
        else:
            print("Using Cosine Annealing With Warm Restarts as LR Scheduler")
            lr_scheduler[node_type] = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer[node_type], T_0=10, T_mult=2)
        # initial_lr_state[node_type] = lr_scheduler[node_type].state_dict()
    if args.gen:
        save_gen_dir = os.path.join(model_dir, "generation")
    # Classification criterion
    # https://arxiv.org/pdf/1901.05555.pdf
    weight = hyperparams['class_weights'].to(args.device)
    criterion_2 = nn.CrossEntropyLoss(reduction='none')
    criterion_1 = ScoreBasedConLoss()  # Use criterion_1._get_name() to get the name of the loss
    # with open(f'{model_dir}/config_{extra_tag}.json', 'w') as fout:
    #     json.dump(hyperparams, fout)
    # criterion_1 = FocalLoss(weight=weight, gamma=0.5, reduction='none')
    # criterion = LDAMLoss(cls_num_list=hyperparams['class_count'], max_m=0.5, s=30, reduction='none').cuda()
    # criterion_2 = SupervisedConLoss(hyperparams['num_classes'])
    #################################
    #           TRAINING            #
    #################################

    print("\n" + bcolors.UNDERLINE + "Trained Model Extra_Tag:" + bcolors.ENDC)
    print(bcolors.OKGREEN + extra_tag + bcolors.ENDC)
    print(bcolors.UNDERLINE + "Class Count:" + bcolors.ENDC)
    print(bcolors.OKGREEN + str(hyperparams['class_count_dic']) + bcolors.ENDC)
    curr_iter_node_type = {node_type: 0 for node_type in train_data_loader.keys()}
    # train_loss_df = pd.DataFrame(columns=['epoch', 'loss'])
    cls_generated, cls_accuracies, cls_losses, losses = [], [], [], []
    start_at = 0
    if args.net_trajectron_ts:
        start_at = int(args.net_trajectron_ts)
    for epoch in range(start_at + 1, start_at + args.train_epochs + 1):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        class_acc = 0
        class_loss = 0
        epoch_loss = 0
        if epoch >= args.warm + 1 and args.gen:
            print("**** Train Epoch with generation ****")
            # Generation process and training with generated data
            train_stats, class_acc, class_loss, class_gen = train_gen_epoch(trajectron, trajectron_g, epoch, curr_iter_node_type, optimizer, lr_scheduler,
                                                                            criterion_2, train_data_loader, hyperparams, log_writer, save_gen_dir, args.device)
            cls_generated.append({"epoch": epoch, "generated per class": class_gen})
        else:
            print("**** Train Epoch without generation ****")
            # if epoch <= 300:
            #     epoch_loss = train_epoch_con_score_based(trajectron, curr_iter_node_type, optimizer, lr_scheduler,
            #                                              criterion_1, train_data_loader, epoch, hyperparams, log_writer, args.device)
            # else:
            class_acc, class_loss = train_epoch(trajectron, curr_iter_node_type, optimizer, lr_scheduler,
                                                criterion_2, train_data_loader, epoch, hyperparams, log_writer,
                                                args.device)
        # if epoch >= 450:
        #     criterion_2 = nn.CrossEntropyLoss(reduction='none', weight=weight)
            # Use now weighted sampler
            # train_data_loader = dict()
            # for node_type_data_set in train_dataset:
            #     node_type_dataloader = utils.data.DataLoader(node_type_data_set,
            #                                                     collate_fn=collate,
            #                                                     pin_memory=False if args.device is 'cpu' else True,
            #                                                     batch_size=args.batch_size,
            #                                                     #  sampler=node_type_data_set.weighted_sampler,
            #                                                     shuffle=True,
            #                                                     num_workers=args.preprocess_workers)
            #     train_data_loader[node_type_data_set.node_type] = node_type_dataloader
            #     # reset lr scheduler
            #     lr_scheduler[node_type_data_set.node_type].load_state_dict(initial_lr_state[node_type])
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            validation_metrics(model=trajectron, criterion=criterion_2,
                               eval_data_loader=eval_data_loader, epoch=epoch,
                               eval_device=args.device, log_writer=log_writer)

        train_dataset.augment = False
        cls_accuracies.append({"epoch": epoch, "accuracy per class": class_acc})
        cls_losses.append({"epoch": epoch, "loss per class": class_loss})
        losses.append({"epoch": epoch, "loss": epoch_loss})
        if (args.save_every is not None and epoch % args.save_every == 0) or epoch == args.train_epochs:
            model_registrar.save_models(epoch, extra_tag)
            with open(f'{model_dir}/cls_accuracies_{extra_tag}.json', 'w') as fout:
                json.dump(cls_accuracies, fout)
            with open(f'{model_dir}/cls_losses_{extra_tag}.json', 'w') as fout:
                json.dump(cls_losses, fout)
            with open(f'{model_dir}/cls_generated_{extra_tag}.json', 'w') as fout:
                json.dump(cls_generated, fout)
            with open(f'{model_dir}/losses_{extra_tag}.json', 'w') as fout:
                json.dump(losses, fout)
