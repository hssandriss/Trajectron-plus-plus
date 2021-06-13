import os
import pathlib
import warnings
from operator import length_hint

import dill
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from numpy.lib.polynomial import polysub
from torch import Size, mode, nn, tensor
from torch.distributions.normal import Normal
from tqdm.auto import tqdm

from model.model_utils import ModeKeys
from model.trajectron_M2m import Trajectron

# warnings.filterwarnings("ignore", category=UserWarning)

# TODO Build an evaluation function that computes the accuracy classwise


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def train_joint_epoch(trajectron, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                      train_data_loader, epoch, top_n, hyperparams, log_writer, device):

    trajectron.model_registrar.train()

    horizon = hyperparams['prediction_horizon']
    coef = hyperparams['main_coef']
    if hyperparams['coef_schedule'] != "":
        coef = 2 * top_n

    if criterion.weight is not None:
        coef = coef * 1e2
    print(f"Using {coef}*CE+EWTA Loss")
    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        loss_epoch = {"classification": [], "regression": [], "joint": []}
        correct_epoch = 0
        class_loss = {k: [] for k in hyperparams['class_count_dic'].keys()}
        class_correct = {k: 0 for k in hyperparams['class_count_dic'].keys()}
        class_count_sampled = {k: 0 for k in hyperparams['class_count_dic'].keys()}
        log_writer.add_scalar(f"{node_type}/classification/train/lr_scheduling",
                              lr_scheduler[node_type].state_dict()['_last_lr'][0], epoch)
        pbar = tqdm(data_loader, ncols=120)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            optimizer[node_type].zero_grad()
            inputs = batch[:-2]
            targets = batch[-2]
            regression_gt = inputs[2].to(device)
            targets = targets.to(device)
            # inputs_shape =[x[i].shape for i in range(5)]
            # some inputs have nan values in x_st_t (inputs[3]) torch.isnan(inputs[3]).sum(2).sum(1).nonzero()
            inputs = trajectron.preprocess_edges(inputs, node_type)
            y_hat, hypothesis, features = trajectron.predict(inputs, horizon, node_type, mode=ModeKeys.TRAIN)
            # if torch.isnan(y_hat).nonzero().sum() > 0:
            #     import pdb; pdb.set_trace()
            train_loss = criterion(y_hat, targets)
            classification_loss = train_loss.mean()
            regression_loss = trajectron.regression_loss(node_type, regression_gt, hypothesis, top_n)
            joint_loss = classification_loss * coef + regression_loss

            pbar.set_description(
                f"Epoch {epoch}, {node_type}, C-L: {classification_loss.item():.2f}, R-L: {regression_loss.item():.2f}, J-L: {joint_loss.item():.2f}")
            joint_loss.backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
            optimizer[node_type].step()
            # Stepping forward the learning rate scheduler and annealers.
            # Per class metrics
            predicted = torch.argmax(F.softmax(y_hat, 1), 1)
            loss_epoch["classification"].append(classification_loss.clone().detach().unsqueeze(0))  # [[bs, 1], [bs, 1]...]
            loss_epoch["regression"].append(regression_loss.clone().detach().unsqueeze(0))  # [[bs, 1], [bs, 1]...]
            loss_epoch["joint"].append(joint_loss.clone().detach().unsqueeze(0))  # [[bs, 1], [bs, 1]...]

            correct_epoch += (predicted == targets).sum().item()
            for k in hyperparams['class_count_dic'].keys():
                k_idx = (targets == k)
                class_count_sampled[k] += k_idx.sum().item()
                if k_idx.sum() == 0:
                    continue
                else:
                    k_loss = train_loss[k_idx].clone().detach()
                    k_targets = targets[k_idx]
                    k_pred = predicted[k_idx]
                    k_correct = (k_targets == k_pred).sum().item()
                    class_loss[k].append(k_loss)
                    class_correct[k] += k_correct
            curr_iter += 1
            lr_scheduler[node_type].step()
        curr_iter_node_type[node_type] = curr_iter
        # Logging
        loss = {}
        for k in loss_epoch.keys():
            loss[k] = torch.cat(loss_epoch[k])
        assert class_count_sampled == hyperparams['class_count_dic'], "You didn't go through all data"
        log_writer.add_scalar(f"{node_type}/classification/train/loss", loss["classification"].mean().log10().item(), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/loss_logarithmic", loss["classification"].mean().item(), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/accuracy",
                              correct_epoch / data_loader.dataset.len, epoch)
        log_writer.add_scalar(f"{node_type}/regression/train/loss", loss["regression"].mean().item(), epoch)
        log_writer.add_scalar(f"{node_type}/joint_training/train/loss", loss["joint"].mean().item(), epoch)
        log_writer.add_scalar(f"{node_type}/joint_training/train/coef", coef, epoch)

        ret_class_acc = {k: class_correct[k] / hyperparams['class_count_dic'][k]
                         for k in hyperparams['class_count_dic'].keys()}
        ret_class_loss_log = {k: torch.cat(class_loss[k]).mean().log10().item()
                              for k in hyperparams['class_count_dic'].keys()}
        ret_class_loss = {k: torch.cat(class_loss[k]).mean().item()
                          for k in hyperparams['class_count_dic'].keys()}
        for k in hyperparams['class_count_dic'].keys():
            log_writer.add_scalar(f"{node_type}/classification/train/loss_class_{k}", ret_class_loss[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification/train/loss_logarithmic_{k}",
                                  ret_class_loss_log[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification/train/accuracy_class_{k}", ret_class_acc[k], epoch)

        print("Epoch Joint Loss: " + bcolors.OKGREEN + str(round(loss["joint"].mean().item(), 3)
                                                           ) + bcolors.ENDC)
        print("Epoch Regression Loss: " + bcolors.OKGREEN + str(round(loss["regression"].mean().item(), 3)
                                                                ) + bcolors.ENDC)
        print("Epoch Classification Loss: " + bcolors.OKGREEN + str(round(loss["classification"].mean().item(), 3)
                                                                    ) + " (" + str(round(loss["classification"].mean().log10().item(), 3)) + ")" + bcolors.ENDC)
        print("Epoch Accuracy: " + bcolors.OKGREEN + str(round(correct_epoch / data_loader.dataset.len, 3)) + bcolors.ENDC)
        print("Accuracy per class: ")
        print(bcolors.OKGREEN + str({k: round(ret_class_acc[k], 3)
                                     for k in hyperparams['class_count_dic'].keys()}) + bcolors.ENDC)
    return ret_class_acc, ret_class_loss


def train_epoch(trajectron, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                train_data_loader, epoch, hyperparams, log_writer, device):

    trajectron.model_registrar.train()
    horizon = hyperparams['prediction_horizon']

    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        loss_epoch = {"classification": []}
        correct_epoch = 0
        class_loss = {k: [] for k in hyperparams['class_count_dic'].keys()}
        class_correct = {k: 0 for k in hyperparams['class_count_dic'].keys()}
        class_count_sampled = {k: 0 for k in hyperparams['class_count_dic'].keys()}
        log_writer.add_scalar(f"{node_type}/classification/train/lr_scheduling",
                              lr_scheduler[node_type].state_dict()['_last_lr'][0], epoch)
        pbar = tqdm(data_loader, ncols=120)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            optimizer[node_type].zero_grad()
            inputs = batch[:-2]
            targets = batch[-2]
            targets = targets.to(device)
            # inputs_shape =[x[i].shape for i in range(5)]
            # some inputs have nan values in x_st_t (inputs[3]) torch.isnan(inputs[3]).sum(2).sum(1).nonzero()
            inputs = trajectron.preprocess_edges(inputs, node_type)
            y_hat, hypothesis, features = trajectron.predict(inputs, horizon, node_type, mode=ModeKeys.TRAIN)
            # if torch.isnan(y_hat).nonzero().sum() > 0:
            #     import pdb; pdb.set_trace()
            train_loss = criterion(y_hat, targets)

            pbar.set_description(
                f"Epoch {epoch}, {node_type}, C-L: {train_loss.mean().item():.2f}")
            train_loss.mean().backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
            optimizer[node_type].step()
            # Stepping forward the learning rate scheduler and annealers.
            # Per class metrics
            predicted = torch.argmax(F.softmax(y_hat, 1), 1)
            loss_epoch["classification"].append(train_loss.mean().clone().detach().unsqueeze(0))  # [[bs, 1], [bs, 1]...]

            correct_epoch += (predicted == targets).sum().item()
            for k in hyperparams['class_count_dic'].keys():
                k_idx = (targets == k)
                class_count_sampled[k] += k_idx.sum().item()
                if k_idx.sum() == 0:
                    continue
                else:
                    k_loss = train_loss[k_idx].clone().detach()
                    k_targets = targets[k_idx]
                    k_pred = predicted[k_idx]
                    k_correct = (k_targets == k_pred).sum().item()
                    class_loss[k].append(k_loss)
                    class_correct[k] += k_correct
            curr_iter += 1
            lr_scheduler[node_type].step()
        curr_iter_node_type[node_type] = curr_iter
        # Logging
        loss = {}
        for k in loss_epoch.keys():
            loss[k] = torch.cat(loss_epoch[k])
        assert class_count_sampled == hyperparams['class_count_dic'], "You didn't go through all data"
        log_writer.add_scalar(f"{node_type}/classification/train/loss", loss["classification"].mean().log10().item(), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/loss_logarithmic", loss["classification"].mean().item(), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/accuracy",
                              correct_epoch / data_loader.dataset.len, epoch)

        ret_class_acc = {k: class_correct[k] / hyperparams['class_count_dic'][k]
                         for k in hyperparams['class_count_dic'].keys()}
        ret_class_loss_log = {k: torch.cat(class_loss[k]).mean().log10().item()
                              for k in hyperparams['class_count_dic'].keys()}
        ret_class_loss = {k: torch.cat(class_loss[k]).mean().item()
                          for k in hyperparams['class_count_dic'].keys()}
        for k in hyperparams['class_count_dic'].keys():
            log_writer.add_scalar(f"{node_type}/classification/train/loss_class_{k}", ret_class_loss[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification/train/loss_logarithmic_{k}",
                                  ret_class_loss_log[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification/train/accuracy_class_{k}", ret_class_acc[k], epoch)

        print("Epoch Classification Loss: " + bcolors.OKGREEN + str(round(loss["classification"].mean().item(), 3)
                                                                    ) + " (" + str(round(loss["classification"].mean().log10().item(), 3)) + ")" + bcolors.ENDC)
        print("Epoch Accuracy: " + bcolors.OKGREEN + str(round(correct_epoch / data_loader.dataset.len, 3)) + bcolors.ENDC)
        print("Accuracy per class:" + bcolors.OKGREEN + str({k: round(ret_class_acc[k], 3)
                                                             for k in hyperparams['class_count_dic'].keys()}) + bcolors.ENDC)
    return ret_class_acc, ret_class_loss


def train_epoch_con(trajectron, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                    train_data_loader, epoch, hyperparams, log_writer, device):
    """
    Training with contrastive loss (discriminate features using class label)
    """
    trajectron.model_registrar.train()

    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        loss_epoch = []
        pos_epoch = []
        neg_epoch = []
        log_writer.add_scalar(f"{node_type}/classification/train/lr_scheduling",
                              lr_scheduler[node_type].state_dict()['_last_lr'][0], epoch)
        pbar = tqdm(data_loader, ncols=120)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            optimizer[node_type].zero_grad()
            inputs = batch[:-2]
            targets = batch[-2]
            targets = targets.to(device)
            inputs = trajectron.preprocess_edges(inputs, node_type)
            _, features = trajectron.predict(inputs, node_type, mode=ModeKeys.TRAIN)
            train_loss, mask_pos, mask_neg = criterion(features, targets)
            pbar.set_description(
                f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f} Positives: {mask_pos.item():.2f}: Negatives: {mask_neg.item():.2f} ")
            # train_loss.register_hook(lambda grad: print(grad))
            train_loss.backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
            optimizer[node_type].step()
            # Stepping forward the learning rate scheduler and annealers.
            loss_epoch.append(train_loss.item())
            neg_epoch.append(mask_neg.item())
            pos_epoch.append(mask_pos.item())
            curr_iter += 1
            lr_scheduler[node_type].step()
        curr_iter_node_type[node_type] = curr_iter
        # Logging
        log_writer.add_scalar(f"{node_type}/classification/train/conloss", np.mean(loss_epoch), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/avg_positive_samples", np.mean(pos_epoch), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/avg_negative_samples", np.mean(neg_epoch), epoch)
        print("Epoch Loss: " + bcolors.OKGREEN + str(round(np.mean(loss_epoch), 3)) + bcolors.ENDC)
    return np.mean(loss_epoch)


def train_epoch_con_score_based(trajectron, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                                train_data_loader, epoch, top_n, hyperparams, log_writer, device):
    """
    Training with contrastive loss (discriminate features using score value)
    """
    trajectron.model_registrar.train()
    horizon = hyperparams['prediction_horizon']
    coef = hyperparams['main_coef']
    if hyperparams['coef_schedule'] != "":
        coef = 2 * top_n
    print(f"Using {coef}*CONLOSS+EWTA Loss")

    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        loss_epoch = []
        pos_epoch = []
        neg_epoch = []
        log_writer.add_scalar(f"{node_type}/classification/train/lr_scheduling",
                              lr_scheduler[node_type].state_dict()['_last_lr'][0], epoch)
        pbar = tqdm(data_loader, ncols=120)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            optimizer[node_type].zero_grad()
            inputs = batch[:-2]
            scores = batch[-1]
            regression_gt = inputs[2].to(device)
            scores = scores.to(device)

            inputs = trajectron.preprocess_edges(inputs, node_type)
            _, hypothesis, features = trajectron.predict(inputs, horizon, node_type, mode=ModeKeys.TRAIN)

            train_loss, mask_pos, mask_neg = criterion(features, scores)
            con_loss = train_loss.mean()
            regression_loss = trajectron.regression_loss(node_type, regression_gt, hypothesis, top_n)
            joint_loss = con_loss * coef + regression_loss
            pbar.set_description(
                f"Epoch {epoch}, {node_type} C-L: {con_loss.item():.2f} <= (P: {mask_pos.item():.2f}: N: {mask_neg.item():.2f}), R-L: {regression_loss.item():.2f}, J-L: {joint_loss.item():.2f}")
            # import pdb; pdb.set_trace()
            # train_loss.register_hook(lambda grad: print(grad))
            train_loss.backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
            optimizer[node_type].step()
            # Stepping forward the learning rate scheduler and annealers.
            loss_epoch.append(train_loss.item())
            neg_epoch.append(mask_neg.item())
            pos_epoch.append(mask_pos.item())
            curr_iter += 1
            lr_scheduler[node_type].step()
        curr_iter_node_type[node_type] = curr_iter
        # Logging
        log_writer.add_scalar(f"{node_type}/classification/train/conloss", np.mean(loss_epoch), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/avg_positive_samples", np.mean(pos_epoch), epoch)
        log_writer.add_scalar(f"{node_type}/classification/train/avg_negative_samples", np.mean(neg_epoch), epoch)
        print("Epoch Loss: " + bcolors.OKGREEN + str(round(np.mean(loss_epoch), 3)) + bcolors.ENDC)
    return np.mean(loss_epoch)


def viz_orig_gen(input_orig,
                 input_gen,
                 log_writer,
                 epoch,
                 save_dir,
                 line_alpha=0.7,
                 line_width=0.2,
                 edge_width=2,
                 circle_edge_width=0.5,
                 node_circle_size=0.3,
                 batch_num=0,
                 kde=False):
    (_, _, _, x_st_t_orig, _, _, _, _) = input_orig
    (_, _, _, x_st_t_gen, _, _, _, _) = input_gen

    plt.clf()
    plt.cla()
    cmap = ['k', 'b', 'y', 'g', 'r']
    x_st_t_orig = x_st_t_orig.squeeze(0)
    x_st_t_gen = x_st_t_gen.squeeze(0)
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.plot(x_st_t_orig[:, 0], x_st_t_orig[:, 1], 'k--')
    ax.text(x_st_t_orig[0, 0], x_st_t_orig[0, 1], "i", fontsize=4)
    ax.text(x_st_t_orig[-1, 0], x_st_t_orig[-1, 1], "f", fontsize=4)
    ax.plot(x_st_t_gen[:, 0], x_st_t_gen[:, 1], 'w--', path_effects=[pe.Stroke(linewidth=edge_width, foreground='k'), pe.Normal()])
    ax.text(x_st_t_gen[0, 0], x_st_t_gen[0, 1], "i", fontsize=4)
    ax.text(x_st_t_gen[-1, 0], x_st_t_gen[-1, 1], "f", fontsize=4)

    angle_diff = (torch.acos(compute_trajectory_angles(x_st_t_gen[:, :2].unsqueeze(0))) -
                  torch.acos(compute_trajectory_angles(x_st_t_orig[:, :2].unsqueeze(0)))).abs().squeeze(0)  # [Ts-2]
    angle_diff = (angle_diff * 180) / 3.14  # to degrees for viz
    j = 0
    for i in range(x_st_t_orig.shape[0] - 2):
        s = round(angle_diff[j].squeeze(0).item(), 3)
        xy = (x_st_t_gen[i + 1][0].item(), x_st_t_gen[i + 1][1].item())
        ax.annotate(text=s, xy=xy, fontsize=4)
        j += 1
    length_origin = compute_trajectory_distance(x_st_t_orig[:, :2].unsqueeze(0))
    length_gen = compute_trajectory_distance(x_st_t_gen[:, :2].unsqueeze(0))
    ax.legend([f'sum angle disp: {round(angle_diff.sum().item(), 3)}',
               f'avg angle disp: {round(angle_diff.mean().item(), 3)}',
               f'original length: {round(length_origin.sum().item(), 3)}',
               f'generated length: {round(length_gen.sum().item(), 3)}'],
              loc=2, prop={'size': 4})
    ax.axis('equal')
    fig.savefig(os.path.join(save_dir, f'example_gen_epoch{epoch}.png'), dpi=300)
    fig.clf()
    plt.close()
    ax.cla()
    if log_writer is not None:
        log_writer.add_figure('generation/trajectory_visualization', fig, epoch)


def input_to_device(input, device):
    """
    TODO: Move batch input to device
    """
    ret = []
    for e in input:
        if isinstance(e, torch.Tensor):
            ret.append(e.to(device))
        elif isinstance(e, dict) and isinstance(list(e.values())[0], list):
            dic = {}
            for k, v in e.items():
                assert all(isinstance(elem, torch.Tensor) for elem in v)
                dic[k] = [tensor.to(device) for tensor in v]
            ret.append(dic)
        elif e is None:
            ret.append(None)
        else:
            ret.append(e)
    return tuple(ret)


def cat_inputs(input_list):
    """
    TODO: Concatenate inputs and targets
    """
    ret = []
    sample_struct = input_list[0]
    for i, e in enumerate(sample_struct):
        if isinstance(e, torch.Tensor):
            ret.append(torch.cat([input[i] for input in input_list], dim=0))
        elif isinstance(e, dict) and isinstance(list(e.values())[0], list):
            dic = {}
            for k, v in e.items():
                assert all(isinstance(elem, torch.Tensor) for elem in v)
                dic[k] = [torch.cat([input[i][k][0] for input in input_list], dim=0),
                          torch.cat([input[i][k][1] for input in input_list], dim=0)]
            ret.append(dic)
        else:
            ret.append(None)
    return tuple(ret)


def detach_batch_input(input):
    # find a way to make this adaptable
    ret = []
    for e in input:
        if isinstance(e, torch.Tensor):
            ret.append(e.detach())
        elif isinstance(e, dict) and isinstance(list(e.values())[0], list):
            dic = {}
            for k, v in e.items():
                assert all(isinstance(elem, torch.Tensor) for elem in v)
                dic[k] = [tensor.detach() for tensor in v]
            ret.append(dic)
        else:
            ret.append(None)
    return tuple(ret)


def clone_batch_input(input):
    # find a way to make this adaptable
    ret = []
    for e in input:
        if isinstance(e, torch.Tensor):
            ret.append(e.clone())
        elif isinstance(e, dict) and isinstance(list(e.values())[0], list):
            dic = {}
            for k, v in e.items():
                assert all(isinstance(elem, torch.Tensor) for elem in v)
                dic[k] = [tensor.clone() for tensor in v]
            ret.append(dic)
        else:
            ret.append(None)
    return ret


def replace_in_batch_input_by_index(input, new_input, indexes):
    assert new_input[3].shape[0] == indexes.shape[0]
    for i, _ in enumerate(input):
        if isinstance(input[i], torch.Tensor):
            input[i][indexes] = new_input[i]
            input[i] = input[i].detach()
            assert input[i].is_leaf
        elif isinstance(input[i], dict):
            for k in input[i].keys():
                assert all(isinstance(elem, torch.Tensor) for elem in input[i][k])
                assert all(isinstance(elem, torch.Tensor) for elem in new_input[i][k])
                input[i][k][0][indexes] = new_input[i][k][0]
                input[i][k][1][indexes] = new_input[i][k][1]
                assert input[i][k][0].is_leaf and input[i][k][1].is_leaf
    return input


def select_from_batch_input(input, select_idx):
    ret = []
    for e in input:
        if isinstance(e, torch.Tensor):
            ret.append(e[select_idx].clone())
        elif isinstance(e, dict) and isinstance(list(e.values())[0], list):
            dic = {}
            for k, v in e.items():
                assert all(isinstance(elem, torch.Tensor) for elem in v)
                dic[k] = [tensor[select_idx].clone() for tensor in v]
            ret.append(dic)
        else:
            ret.append(None)
    return ret


def append_to_batch_input(input, new_input):
    ret = []
    for idx, e in enumerate(input):
        if isinstance(e, torch.Tensor):
            ret.append(torch.cat((e, new_input[idx]), dim=0))
        elif isinstance(e, dict) and isinstance(list(e.values())[0], list):
            dic = {}
            for k, v in e.items():
                assert all(isinstance(elem, torch.Tensor) for elem in v)
                dic[k] = [torch.cat((tensor, new_input[idx][k][i]), dim=0) for i, tensor in enumerate(v)]
            ret.append(dic)
        else:
            ret.append(None)
    return ret


def validation_metrics(model, criterion, eval_data_loader, epoch, eval_device, hyperparams, log_writer):
    model.model_registrar.eval()
    horizon = hyperparams['prediction_horizon']
    with torch.no_grad():
        loss = {}
        accuracy = {}
        # Calculate evaluation loss
        for node_type, data_loader in eval_data_loader.items():
            eval_loss = {"regression": [], "classification": []}
            correct = 0
            num_samples = 0
            print(f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
            pbar = tqdm(data_loader, ncols=120)
            for batch in pbar:
                inputs = batch[:-2]
                targets = batch[-2]
                targets = targets.to(eval_device)
                regression_gt = inputs[2].to(eval_device)
                inputs = model.preprocess_edges(inputs, node_type)
                inputs = input_to_device(inputs, eval_device)
                y_hat, hypothesis, _ = model.predict(inputs, horizon, node_type, mode=ModeKeys.EVAL)
                curr_closs = criterion(y_hat, targets).mean()
                curr_rloss = model.regression_loss(node_type, regression_gt, hypothesis, top_n=1)
                eval_loss["classification"].append(curr_closs.item())
                eval_loss["regression"].append(curr_rloss.item())
                predicted = torch.argmax(F.softmax(y_hat, 1), 1)
                num_samples += targets.shape[0]
                correct += (predicted == targets).sum().item()
                pbar.set_description(
                    f"Epoch {epoch}, {node_type} C-L: {curr_closs:.2f}, Acc: {(correct/num_samples):.2f}, R-L: {curr_rloss:.2f} ")
                del batch
            loss[node_type] = {k: round(np.mean(eval_loss[k]), 3) for k in eval_loss.keys()}
            accuracy[node_type] = round(np.sum(correct) / num_samples, 3)
            log_writer.add_scalar(f"{node_type}/joint_training/validation/accuracy", accuracy[node_type], epoch)
            log_writer.add_scalar(f"{node_type}/joint_training/validation/classification_loss", loss[node_type]["classification"], epoch)
            log_writer.add_scalar(f"{node_type}/joint_training/validation/regression_loss", loss[node_type]["regression"], epoch)
            print(f"C-L: {loss[node_type]['classification']}, Accuracy {accuracy[node_type]}, R-L: {loss[node_type]['regression']}")
    # import pdb; pdb.set_trace()
    return loss, accuracy


def classwise_loss(outputs, targets):
    """
     Returns logit confidence
    """
    out_1hot = torch.zeros_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), 1)
    return (outputs * out_1hot).sum(1).mean()


def make_step(grad, attack, step_size):
    if attack == 'l2':
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=1).view(-1, 1, 1)
        scaled_grad = grad / (grad_norm + 1e-10)
        step = step_size * scaled_grad
    elif attack == 'inf':
        step = step_size * torch.sign(grad)
    else:
        step = step_size * grad
    return step


def random_perturb(inputs, attack, eps=0.5, std=0.001):
    if attack == 'inf':
        r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
    elif attack == 'normal':
        # r_inputs = torch.normal(0, 0.1, size=inputs.shape)
        r_inputs = std * torch.randn(size=inputs.shape)
    else:
        r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
    return r_inputs


def sum_tensor(tensor):
    return tensor.float().sum().item()


def filter_unreliable(trajectron_g, node_type, threshold, seed_targets, seed_inputs):
    outputs_g, _, _ = trajectron_g.predict(seed_inputs, 1, node_type, mode=ModeKeys.EVAL)
    one_hot = torch.zeros_like(outputs_g)
    one_hot.scatter_(1, seed_targets.view(-1, 1), 1)
    probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]
    mask_reliable = (probs_g >= threshold)
    return mask_reliable


def compute_squared_dist_xy(x1, x2, dim):
    return (((x2 - x1)**2).sum(dim=dim))**0.5


def compute_trajectory_distance(x):
    """
    :input: Expects shape [Bs, Ts, 2]
    :returns: angles of trajectory [Bs, Ts-1]
    """
    # with torch.autograd.set_detect_anomaly(True):
    # x = replace_nan(x)
    length = []
    for i in range(x.shape[1] - 1):
        u_tensor = (x[:, i + 1, :] - x[:, i, :])
        length.append(torch.norm(u_tensor, p=2, dim=1).unsqueeze(0))
    return torch.cat(length, dim=1)


def compute_trajectory_angles(x, epsilon=1e-7):
    """
    :input: Expects shape [Bs, Ts, 2]
    :returns: Cos of angles of trajectory [Bs, Ts-2]
    """
    # with torch.autograd.set_detect_anomaly(True):
    # x = replace_nan(x)
    angles = []
    for i in range(x.shape[1] - 2):
        u_tensor = (x[:, i + 1, :] - x[:, i, :])
        v_tensor = (x[:, i + 2, :] - x[:, i + 1, :])
        cos = torch.clamp(F.cosine_similarity(u_tensor, v_tensor, dim=1), -1 + epsilon, 1 - epsilon)
        # when saving angles using acos: Use clamp https://github.com/pytorch/pytorch/issues/8069
        # angles.append(torch.acos(cos).unsqueeze(1)) # can be unbound
        angles.append(cos.unsqueeze(1))  # [-1, 1]
    return torch.cat(angles, dim=1)


def replace_nan(tensor, val=0):
    tensor[tensor != tensor] = val
    return tensor


def mask_nan(x):
    """"
    Args:
        x (Tensor: [Bs, Ts, 6]): 
    """
    liste = x.split(dim=0, split_size=1)
    maske = [(~torch.any(torch.isnan(t))).unsqueeze(0) for t in liste]
    return torch.cat(maske, dim=0)


def generation(trajectron_g, trajectron, node_type, device, seed_inputs, seed_targets, gen_targets, p_accept,
               gamma, lam, step_size, hyperparams, random_start=True, max_iter=10):
    """
    Over-sampling via M2m Algorithm (Pg: 4) from line 7:e

    Args:
        inputs: seed input x0
        seed_targets: source target k0
        targets: goal target k*

    Returns:
        inputs: generated input x*
        correct: if the generated input is good enough to be from k* or should
                 we choose a random sample from k* (oversampling)
    """
    torch.backends.cudnn.enabled = False
    trajectron.model_registrar.eval()
    trajectron_g.model_registrar.eval()
    criterion = nn.CrossEntropyLoss()
    gen_coef = hyperparams['gen_coef']
    gen_kl_coef = hyperparams['gen_kl_coef']
    edge_gen = False
    if hyperparams['gen_edges'] == 'yes':
        edge_gen = True
    # How well does g perform initially on seed_inputs/targets ?
    mask_reliable = filter_unreliable(trajectron_g, node_type, 0.8, seed_targets, seed_inputs)
    # We will use x_st_t and preprocessed_edges
    (first_history_index, x_t, y_t, x_st_t, y_st_t, preprocessed_edges, robot_traj_st_t, map) = seed_inputs
    if not first_history_index.is_leaf or first_history_index.requires_grad:
        (first_history_index, x_t, y_t, y_st_t, robot_traj_st_t, map) = tuple(tensor.detach().requires_grad_(False) if isinstance(tensor, torch.Tensor) else None
                                                                              for tensor in (first_history_index, x_t, y_t, y_st_t, robot_traj_st_t, map))
        for key in preprocessed_edges.keys():  # Edge data
            preprocessed_edges[key][0] = preprocessed_edges[key][0].detach().requires_grad_(False)
            preprocessed_edges[key][1] = preprocessed_edges[key][1].detach().requires_grad_(False)
    # ! Preparing state history while keeping initial and final states fixed
    x_st_t_i = x_st_t[:, 0, :].unsqueeze(1).to(device)
    x_st_t_f = x_st_t[:, -1, :].unsqueeze(1).to(device)
    x_st_t_o = x_st_t[:, 1:-1, :].to(device)
    if edge_gen:
        # ! Preparing edges (decomposing the preprocessed edges dict)
        edge_keys = list(preprocessed_edges.keys())
        edge_masks = [preprocessed_edges[k][1] for k in preprocessed_edges.keys()]
        combined_neighbors = [preprocessed_edges[k][0][:, :, :6] for k in preprocessed_edges.keys()]  # [bs, hist_len, state]
    # * Random Noise
    if random_start:
        # ! Adding noise just in the middle steps in node history
        random_noise = random_perturb(x_st_t_o, 'normal').to(device)
        x_st_t_o = torch.clamp(x_st_t_o + random_noise, 0, 1)
        # ! Adding noise to combined neighbors
        if edge_gen:
            for i in range(len(combined_neighbors)):
                # random_noise = random_perturb(combined_neighbors[i], 'l2', 0.5)
                random_noise = random_perturb(combined_neighbors[i], 'normal').to(device)
                combined_neighbors[i] = torch.clamp(combined_neighbors[i] + random_noise, 0, 1)
    # For Verification
    initial_rnn_weights_g = trajectron_g.model_registrar.get_name_match(
        "PEDESTRIAN/node_history_encoder")._modules["0"].weight_ih_l0.clone().data
    initial_rnn_weights_f = trajectron.model_registrar.get_name_match(
        "PEDESTRIAN/node_history_encoder")._modules["0"].weight_ih_l0.clone().data
    # ! Saving initial data
    x_st_t_o_pre = x_st_t_o.clone().detach().requires_grad_(False)
    x_st_t_pre = x_st_t.clone().detach().requires_grad_(False)
    if edge_gen:
        combined_neighbors_pre = combined_neighbors[0].clone().detach().requires_grad_(False)
    # ! Loop for optimizing the objective
    for _ in range(max_iter):
        # setting requires grad to True
        x_st_t_o = x_st_t_o.clone().detach().requires_grad_(True)
        x_st_t = torch.cat((x_st_t_i, x_st_t_o, x_st_t_f), dim=1).to(device)
        if edge_gen:
            combined_neighbors = [tensor.clone().detach().requires_grad_(True) for tensor in combined_neighbors]
            joint_histories = [torch.cat((x_st_t, tensor), dim=-1) for tensor in combined_neighbors]
            edge_values = [[joint_histories[i], edge_masks[i]] for i in range(len(edge_masks))]
            preprocessed_edges = dict(zip(edge_keys, edge_values))
        inputs = (first_history_index, x_t, y_t, x_st_t, y_st_t, preprocessed_edges, robot_traj_st_t, map)
        # Forward Pass
        outputs_g, _, _ = trajectron_g.predict(inputs, 1, node_type, mode=ModeKeys.EVAL)
        outputs_r, _, _ = trajectron.predict(inputs, 1, node_type, mode=ModeKeys.EVAL)
        # M2m original objective
        loss = criterion(outputs_g, gen_targets) + lam * classwise_loss(outputs_r, seed_targets)
        if hyperparams['gen_angular_obj'] == 'yes':
            angle_objective = (compute_trajectory_angles(x_st_t[:, :, :2]) - compute_trajectory_angles(x_st_t_pre[:, :, :2])).pow(2).mean()
            loss += gen_coef * angle_objective
        if hyperparams['gen_distance_obj'] == 'yes':
            distance_objective = (compute_trajectory_distance(x_st_t[:, :, :2]) - compute_trajectory_distance(x_st_t_pre[:, :, :2])).pow(2).mean()
            loss += gen_coef * distance_objective
        if hyperparams['gen_kl_obj'] == 'yes':
            disp = (((x_st_t[:, :, :2] - x_st_t_pre[:, :, :2])**2).sum(dim=2))**0.5
            scales = disp.mean(1).log10().int().clamp_(-6, 6).float()  # Scales per sample
            std = (10**scales).unsqueeze(1) * torch.ones(disp.shape).to(device)
            mean = torch.zeros(disp.shape).to(device)
            normal_dist = Normal(mean, std)
            noise_samples = normal_dist.sample()
            prob_noise = F.softmax(input=noise_samples, dim=-1)
            log_prob_disp = F.log_softmax(input=disp, dim=-1)
            kl_objective = F.kl_div(log_prob_disp, prob_noise, reduction='mean')
            if kl_objective < 0:
                print("Instability! KL divergence should be always positive")
                import pdb; pdb.set_trace()
            loss += gen_kl_coef * kl_objective
        # Computing gradient wrt the input
        if edge_gen:
            grad_vals = torch.autograd.grad(loss, [x_st_t_o, *combined_neighbors])
        else:
            grad_vals = torch.autograd.grad(loss, [x_st_t_o])
        grad_x_st_t_o = grad_vals[0]
        x_st_t_o = x_st_t_o - make_step(grad_x_st_t_o, 'l2', step_size)
        x_st_t_o = torch.clamp(x_st_t_o, 0, 1)
        if edge_gen:
            grad_neighbors = grad_vals[1:]
            for i in range(len(combined_neighbors)):
                combined_neighbors[i] = combined_neighbors[i] - make_step(grad_neighbors[i], 'l2', step_size)
                combined_neighbors[i] = torch.clamp(combined_neighbors[i], 0, 1)
    # Verification that weights of the network remain unchanged during generation loop
    after_rnn_weights_g = trajectron_g.model_registrar.get_name_match("PEDESTRIAN/node_history_encoder")._modules["0"].weight_ih_l0.clone()
    after_rnn_weights_f = trajectron.model_registrar.get_name_match("PEDESTRIAN/node_history_encoder")._modules["0"].weight_ih_l0.clone()
    assert torch.all(initial_rnn_weights_g.eq(after_rnn_weights_g))
    assert torch.all(initial_rnn_weights_f.eq(after_rnn_weights_f))
    # ! Recombining the new input
    x_st_t = torch.cat((x_st_t_i, x_st_t_o, x_st_t_f), dim=1).to(device)
    if edge_gen:
        joint_histories = [torch.cat((x_st_t, tensor), dim=-1).to(device) for tensor in combined_neighbors]
        edge_values = [[joint_histories[i], edge_masks[i]] for i in range(len(edge_masks))]
        preprocessed_edges = dict(zip(edge_keys, edge_values))
    inputs = (first_history_index, x_t, y_t, x_st_t, y_st_t, preprocessed_edges, robot_traj_st_t, map)
    inputs = detach_batch_input(inputs)
    # ! Acceptance criterion
    outputs_g, _, _ = trajectron_g.predict(inputs, 1, node_type, mode=ModeKeys.EVAL)
    one_hot = torch.zeros_like(outputs_g)
    one_hot.scatter_(1, gen_targets.view(-1, 1), 1)
    probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]
    correct = ((probs_g >= gamma) * torch.bernoulli(p_accept)).type(torch.bool).to(device)
    correct = correct * mask_reliable
    torch.backends.cudnn.enabled = True
    trajectron.model_registrar.train()
    return inputs, correct


def train_net(trajectron, trajectron_g, node_type, criterion, coef, optimizer, lr_scheduler, inputs_orig_tuple,
              targets_orig, gen_idx, gen_targets, top_n, hyperparams, device):

    horizon = hyperparams['prediction_horizon']

    class_gen_batch = {k: 0 for k in hyperparams['class_count_dic'].keys()}
    class_loss_batch = {k: 0 for k in hyperparams['class_count_dic'].keys()}
    class_acc_batch = {k: 0 for k in hyperparams['class_count_dic'].keys()}
    ########################
    batch_size = inputs_orig_tuple[0].size(0)
    ########################
    # inputs = inputs_orig_tuple.clone()
    # [tensor.is_cuda if isinstance(tensor, torch.Tensor) else None for tensor in inputs_orig_tuple]
    inputs_tuple = clone_batch_input(inputs_orig_tuple)

    targets = targets_orig.clone()
    ########################
    N_SAMPLES_PER_CLASS_T = torch.Tensor(hyperparams['class_count']).to(device)
    N_CLASSES = len(hyperparams['class_count'])
    ########################
    bs = N_SAMPLES_PER_CLASS_T[targets_orig].repeat(gen_idx.size(0), 1)  # [gen_idx, BS]
    gs = N_SAMPLES_PER_CLASS_T[gen_targets].view(-1, 1)
    delta = F.relu(bs - gs)
    p_accept = 1 - hyperparams['beta'] ** delta
    mask_valid = (p_accept.sum(1) > 0)

    gen_idx = gen_idx[mask_valid]
    gen_targets = gen_targets[mask_valid]
    p_accept = p_accept[mask_valid]  # [gen_idx, BS]
    # Sample column indices (BS) from p_accept for each row using probs defined in each row of p_accept
    select_idx = torch.multinomial(p_accept, 1, replacement=True).view(-1)
    # Select the chosen column indices (select_idx) from p_accept
    p_accept = p_accept.gather(1, select_idx.view(-1, 1)).view(-1)
    # The selected columns are going to be used as seed i.e. x0, k0 in the current batch
    seed_targets = targets_orig[select_idx]
    # seed_inputs = inputs_orig_tuple[select_idx]
    seed_inputs = select_from_batch_input(inputs_tuple, select_idx)
    # Masking indices that already have nan values in x_st_t
    mask_nans = mask_nan(seed_inputs[3])
    seed_inputs = select_from_batch_input(seed_inputs, mask_nans)
    seed_targets = seed_targets[mask_nans]
    gen_targets = gen_targets[mask_nans]
    p_accept = p_accept[mask_nans]
    gen_idx = gen_idx[mask_nans]
    assert torch.any(torch.isnan(seed_inputs[3])).item() == False, "NaN gradient risk in generation process"
    # ! Now we have sampled seed classes k0 of initial point x0 given gen_target class k.
    gen_inputs, correct_mask = generation(trajectron_g, trajectron, node_type, device, seed_inputs, seed_targets, gen_targets,
                                          p_accept, hyperparams['gamma'], hyperparams['lam'], hyperparams['step_size'], hyperparams, True, hyperparams['attack_iter'])
    num_gen = sum_tensor(correct_mask)

    # TODO Add possibility to extend the batch with generated samples
    if hyperparams['append_gen'] == 'no':
        num_others = batch_size - num_gen
        #######################
        # Only change the input of correctly generated samples
        gen_c_idx = gen_idx[correct_mask]
        others_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        others_mask[gen_c_idx] = 0  # can be used for computing regression loss
        others_idx = others_mask.nonzero().view(-1)
        if num_gen > 0:
            gen_inputs_c = select_from_batch_input(gen_inputs, correct_mask)
            new_inputs = replace_in_batch_input_by_index(inputs_tuple, gen_inputs_c, gen_c_idx)
            new_targets = targets.clone()
            gen_targets_c = gen_targets[correct_mask]
            new_targets[gen_c_idx] = gen_targets_c
            inputs_tuple = new_inputs
    else:
        #######################
        # Here we want to add the new generated samples to the batch
        # gen_inputs, gen_targets
        others_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
        others_idx = others_mask.nonzero().view(-1)
        gen_c_idx = (others_mask == 0).nonzero().view(-1)
        num_others = batch_size
        if num_gen > 0:
            others_mask = torch.cat((others_mask, torch.zeros(int(num_gen), dtype=torch.bool, device=device)), dim=0)
            others_idx = others_mask.nonzero().view(-1)
            gen_c_idx = (others_mask == 0).nonzero().view(-1)
            num_others = batch_size
            gen_inputs_c = select_from_batch_input(gen_inputs, correct_mask)
            gen_targets_c = gen_targets[correct_mask]
            inputs_tuple = append_to_batch_input(inputs_tuple, gen_inputs_c)
            targets = torch.cat((targets, gen_targets_c), dim=0)
            assert targets.shape[0] > batch_size
    # Normal training for a minibatch
    regression_gt = inputs_tuple[2].to(device)
    optimizer[node_type].zero_grad()
    y_hat, hypothesis, _ = trajectron.predict(inputs_tuple, horizon, node_type, mode=ModeKeys.TRAIN)
    train_loss = criterion(y_hat, targets)
    classification_loss = train_loss.mean()
    regression_loss = trajectron.regression_loss(node_type, regression_gt[others_mask], hypothesis[others_mask], top_n)
    joint_loss = classification_loss * coef + regression_loss
    joint_loss.backward()
    # Clipping gradients.
    if hyperparams['grad_clip'] is not None:
        nn.utils.clip_grad_value_(trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
    optimizer[node_type].step()
    ################################
    # Summing up the class based gens, loss, acc for current batch
    ################################
    predicted = torch.argmax(F.softmax(y_hat, 1), 1)
    for k in hyperparams['class_count_dic'].keys():
        k_idx = (targets == k)
        if k_idx.sum() == 0:
            continue
        else:
            k_loss = train_loss[k_idx].mean().item()
            k_total = k_idx.sum().item()
            k_targets = targets[k_idx]
            k_pred = predicted[k_idx]
            k_correct = (k_targets == k_pred).sum().item()
            k_acc = k_correct / k_total
            class_loss_batch[k] += k_loss
            class_acc_batch[k] += k_acc
            if num_gen > 0:
                class_gen_batch[k] += (gen_targets_c == k).sum().item()
    ################################
    # For logging the training
    ################################
    oth_loss_total = sum_tensor(train_loss[others_idx])
    gen_loss_total = sum_tensor(train_loss[gen_c_idx])

    _, predicted = torch.max(y_hat[others_idx].data, 1)
    num_correct_oth = sum_tensor(predicted.eq(targets[others_idx]))

    num_correct_gen, p_g_orig, p_g_targ = 0, 0, 0
    success = torch.zeros(N_CLASSES, 2)

    if num_gen > 0:
        _, predicted_gen = torch.max(y_hat[gen_c_idx].data, 1)
        num_correct_gen = sum_tensor(predicted_gen.eq(targets[gen_c_idx]))
        probs = torch.softmax(y_hat[gen_c_idx], 1).data

        p_g_orig = probs.gather(1, seed_targets[correct_mask].view(-1, 1))
        p_g_orig = sum_tensor(p_g_orig)

        p_g_targ = probs.gather(1, gen_targets_c.view(-1, 1))
        p_g_targ = sum_tensor(p_g_targ)

    for i in range(N_CLASSES):
        if num_gen > 0:
            success[i, 0] = sum_tensor(gen_targets_c == i)
        success[i, 1] = sum_tensor(gen_targets == i)

    batch_gen_inputs = None
    batch_gen_outputs = None
    batch_orig_inputs = None
    batch_orig_outputs = None
    if num_gen > 0:
        batch_orig_inputs = select_from_batch_input(seed_inputs, correct_mask)
        batch_orig_outputs = seed_targets[correct_mask]
        batch_gen_inputs = gen_inputs_c
        batch_gen_outputs = gen_targets_c
    return classification_loss, regression_loss, joint_loss, oth_loss_total, gen_loss_total, num_others, num_correct_oth, num_gen, num_correct_gen, p_g_orig, p_g_targ, \
        success, class_gen_batch, class_loss_batch, class_acc_batch, batch_gen_inputs, batch_gen_outputs, batch_orig_inputs, batch_orig_outputs


def train_gen_epoch(trajectron, trajectron_g, epoch, top_n, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                    train_data_loader, hyperparams, log_writer, save_gen_dir, device):

    N_SAMPLES_PER_CLASS_T = torch.Tensor(hyperparams['class_count']).to(device)
    trajectron_g.model_registrar.eval()
    trajectron.model_registrar.train()

    results = dict()

    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        coef = hyperparams['main_coef']
        if hyperparams['coef_schedule'] != "":
            coef = 2 * top_n
        if criterion.weight is not None:
            coef = coef * 1e2
        print(f"Using {coef}*CE+EWTA Loss")
        oth_loss, gen_loss = 0, 0
        correct_oth = 0
        correct_gen = 0
        total_oth, total_gen = 1e-6, 1e-6
        p_g_orig, p_g_targ = 0, 0
        t_success = torch.zeros(hyperparams['num_classes'], 2)
        class_gen = {k: [] for k in hyperparams['class_count_dic'].keys()}
        class_loss = {k: [] for k in hyperparams['class_count_dic'].keys()}
        class_acc = {k: [] for k in hyperparams['class_count_dic'].keys()}
        loss_epoch = {"classification": [], "regression": [], "joint": []}
        correct_epoch = 0
        gen_ins = []
        gen_outs = []
        orig_ins = []
        orig_outs = []
        pbar = tqdm(data_loader, ncols=180, position=0, leave=True)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            inputs = batch[:-2]
            targets = batch[-2]
            targets = targets.to(device)
            inputs = trajectron.preprocess_edges(inputs, node_type)
            inputs = input_to_device(inputs, device)
            # Set a generation target for current batch with re-sampling
            # Keep the sample with this probability
            gen_probs = N_SAMPLES_PER_CLASS_T[targets] / N_SAMPLES_PER_CLASS_T[0]
            # Here choose randomly possible gen index according to prob gen_prob choose 1 and 0
            gen_index = (1 - torch.bernoulli(gen_probs)).nonzero()
            gen_index = gen_index.view(-1)
            gen_targets = targets[gen_index]
            original_count_stats = {k: (targets == k).sum().item() for k in range(hyperparams['num_classes'])}
            classification_loss, regression_loss, joint_loss, t_loss, g_loss, num_others, num_correct, num_gen, num_gen_correct, p_g_orig_batch, p_g_targ_batch, success, class_gen_batch, class_loss_batch, class_acc_batch, gen_in, gen_out, orig_in, orig_out = train_net(
                trajectron, trajectron_g, node_type, criterion, coef, optimizer, lr_scheduler, inputs, targets, gen_index, gen_targets, top_n, hyperparams, device)
            # Count for the modified batch
            loss_epoch["classification"].append(classification_loss.unsqueeze(0))
            loss_epoch["regression"].append(regression_loss.unsqueeze(0))
            loss_epoch["joint"].append(joint_loss.unsqueeze(0))
            gen_stats = " ".join([f"{k}: {class_gen_batch[k]}/{original_count_stats[k]}" for k in range(hyperparams['num_classes'])])
            # gen_stats = sum([class_gen_batch[k] for k in class_gen_batch.keys()])
            pbar.set_description(
                f"Epoch: {epoch}, {node_type}, Gen: ({gen_stats}), C-L: {classification_loss:.2f} (Oth: {t_loss/num_others+1e-6:.2f} - Gen: {g_loss/(num_gen+1e-6):.2f}), R-L: {regression_loss:.2f}, J-L: {joint_loss:.2f}")
            if gen_in is not None:
                # to CPU in order to save memory
                gen_ins.append(input_to_device(gen_in, 'cpu'))
                gen_outs.append(gen_out.to('cpu'))
                orig_ins.append(input_to_device(orig_in, 'cpu'))
                orig_outs.append(orig_out.to('cpu'))
            for k in hyperparams['class_count_dic'].keys():
                class_gen[k].append(class_gen_batch[k])
                class_loss[k].append(class_loss_batch[k])
                class_acc[k].append(class_acc_batch[k])
            oth_loss += t_loss  # Loss for the other samples
            gen_loss += g_loss  # loss for gen samples
            total_oth += num_others  # not generated
            correct_oth += num_correct  # Adding to above, the model predicts them correctly.
            total_gen += num_gen  # correctly generated.
            correct_gen += num_gen_correct  # Adding to above, the model predicts them correctly.
            p_g_orig += p_g_orig_batch  # logits confidence on the original label
            p_g_targ += p_g_targ_batch  # logits confidence on the target label
            t_success += success
            correct_epoch += correct_gen + correct_oth
            curr_iter += 1
            # Stepping forward the learning rate scheduler and annealers.
            lr_scheduler[node_type].step()
        # Saving generated data
        if total_gen > 1e-6:
            # import pdb; pdb.set_trace()
            print("Generated data !")
            # Concatenate inputs
            gen_o = torch.cat(gen_outs, dim=0)
            gen_i = cat_inputs(gen_ins)
            orig_o = torch.cat(orig_outs, dim=0)
            orig_i = cat_inputs(orig_ins)
            assert gen_o.shape[0] == gen_i[0].shape[0]
            assert orig_o.shape[0] == orig_i[0].shape[0]
            file = {
                "original":
                {
                    "inputs": orig_i,
                    "outputs": orig_o
                },
                "generated":
                {
                    "inputs": gen_i,
                    "outputs": gen_o
                },
            }
            # Saving to dill pkl file
            print("Saving generated data...")
            pathlib.Path(save_gen_dir).mkdir(parents=True, exist_ok=True)
            file_path = os.path.join(save_gen_dir, f"data_orig_gen_{epoch}.pkl")
            with open(file_path, "wb") as dill_file:
                dill.dump(file, dill_file)
            # sample 1 examples:
            sample_indices = torch.randperm(gen_o.shape[0])[:1]
            orig_sample = select_from_batch_input(orig_i, sample_indices)
            gen_sample = select_from_batch_input(gen_i, sample_indices)
            # Drawing random trajectories
            viz_orig_gen(orig_sample, gen_sample, log_writer, epoch, save_dir=save_gen_dir)

        results[node_type] = {
            'train_loss': oth_loss / total_oth,
            'gen_loss': gen_loss / total_gen,
            'train_acc': 100. * correct_oth / total_oth,
            'gen_acc': 100. * correct_gen / total_gen,
            'p_g_orig': p_g_orig / total_gen,
            'p_g_targ': p_g_targ / total_gen,
            't_success': t_success,
            'acc': correct_epoch / len(data_loader.dataset)
        }
        # Logging
        loss = {}
        for k in loss_epoch.keys():
            loss[k] = torch.cat(loss_epoch[k])

        log_writer.add_scalar(f"{node_type}/classification_f/train/regression_loss", loss["regression"].mean(), epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/classification_loss", loss["classification"].mean(), epoch)
        log_writer.add_scalar(f"{node_type}/joint_training/train/loss", loss["joint"].mean(), epoch)
        # log_writer.add_scalar(f"{node_type}/joint_training/train/coef", coef, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/oth_loss", oth_loss / total_oth, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/gen_loss", gen_loss / total_gen, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/train_acc", 100 * results[node_type]['acc'], epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/oth_acc", 100. * correct_oth / total_oth, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/gen_acc", 100. * correct_gen / total_gen, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/p_g_orig", p_g_orig / total_gen, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/p_g_targ", p_g_targ / total_gen, epoch)

        msg = '%s | t_Loss: %.3f | g_Loss: %.3f | Acc: %.3f (%d/%d) | Acc_gen: %.3f (%d/%d) ' \
            '| Prob_orig: %.3f | Prob_targ: %.3f' % (str(node_type),
                                                     results[node_type]['train_loss'], results[node_type]['gen_loss'],
                                                     results[node_type]['train_acc'], correct_oth, total_oth,
                                                     results[node_type]['gen_acc'], correct_gen, total_gen,
                                                     results[node_type]['p_g_orig'], results[node_type]['p_g_targ']
                                                     )
        ret_class_gen = {k: int(np.sum(class_gen[k])) for k in hyperparams['class_count_dic'].keys()}
        ret_class_acc = {k: round(np.mean(class_acc[k]), 3) for k in hyperparams['class_count_dic'].keys()}
        ret_class_loss = {k: round(np.mean(class_loss[k]), 3) for k in hyperparams['class_count_dic'].keys()}
        for k in hyperparams['class_count_dic'].keys():
            log_writer.add_scalar(f"{node_type}/classification_f/train/loss_class_{k}", ret_class_loss[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification_f/train/accuracy_class_{k}", ret_class_acc[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification_f/train/gen_class_{k}", ret_class_gen[k], epoch)

        print(bcolors.OKGREEN + msg + bcolors.ENDC)

        print(bcolors.UNDERLINE + "Overall classification loss:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(round(loss["classification"].mean().item(), 3)) + bcolors.ENDC)
        print(bcolors.UNDERLINE + "Overall regression loss:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(round(loss["regression"].mean().item(), 3)) + bcolors.ENDC)
        print(bcolors.UNDERLINE + "Overall joint loss:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(round(loss["joint"].mean().item(), 3)) + bcolors.ENDC)

        print(bcolors.UNDERLINE + "Class Gens:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(ret_class_gen) + bcolors.ENDC)
        print(bcolors.UNDERLINE + "Class Loss:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(ret_class_loss) + bcolors.ENDC)
        print(bcolors.UNDERLINE + "Class Acc:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(ret_class_acc) + bcolors.ENDC)
        print()
        # import pdb; pdb.set_trace()

    return results, ret_class_acc, ret_class_loss, ret_class_gen


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, weights=None, max_m=0.5, s=30, reduction='none'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))  # list of deltas without cste C
        m_list = m_list * (max_m / np.max(m_list))  # list of deltas when we multiply by C
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list  # inverse proportional to cls_num_list ==> sum != 1
        assert s > 0
        self.s = s
        self.cls_num_list = cls_num_list  # how many observation per class
        self.reduction = reduction
        self.weights = weights
        # majority to minority

    def forward(self, x, target):
        # x shape: [bs, nb_classes]
        index = torch.zeros_like(x, dtype=torch.uint8)  # zeros [bs, nb_classes]
        index.scatter_(1, target.data.view(-1, 1), 1)  # put one in the correct class

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))  # [bs, 1] Delta of correct class
        x_m = x - batch_m  # prediction of ALL classees - Delta

        output = torch.where(index, x_m, x)  # we take x_m if index else we take x
        return F.cross_entropy(self.s * output, target, weight=self.weights, reduction=self.reduction)


class SupervisedConLoss(nn.Module):
    def __init__(self, num_classes, base_temperature=0.07):
        super(SupervisedConLoss, self).__init__()
        self.base_temperature = base_temperature
        self.num_classes = num_classes

    def forward(self, features, targets, temp=0.1):
        """Calculates supervised contrastive loss based on label

        Args:
            features (torch.Tensor): [bs, feature_size]
            targets (torch.Tensor): [bs]
            temp (float, optional): [description]. Defaults to 0.1.

        Returns:
            loss : loss mean
            avg_num_positives per sample
            avg_num_negatives_per sample
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        bs = features.shape[0]
        targets_one_hot = torch.zeros(size=(bs, self.num_classes)).to(device)
        targets_one_hot.scatter_(1, targets.view(-1, 1), 1)
        mask_anchor = torch.eye(n=bs).to(device)
        mask_positives = torch.matmul(targets_one_hot, targets_one_hot.T)  # [bs, bs]
        mask_negatives = torch.ones(size=(bs, bs)).to(device) - mask_positives  # [bs, bs]
        mask_positives = mask_positives - mask_anchor
        logits_mask = mask_positives + mask_negatives
        # Now we have computed all masks without self-contrast

        # compute logits
        logits = torch.div(torch.matmul(features, features.T), temp)  # [bs, bs]

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)  # (bs,1)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)  # [bs, bs] - [bs, 1]
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)

        # loss
        loss = - (temp / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, bs).mean()

        return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1

    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1
        y_true = F.one_hot(y_true, 2).to(torch.float32)
        y_pred = F.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1 - self.epsilon)
        return 1 - f1.mean(), precision, recall


def focal_loss(input_values, gamma):
    """Computes the focal loss
    Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py
    """
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values
    return loss


class FocalLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""

    def __init__(self, weight=None, gamma=0., reduction='mean'):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, weight=self.weight, reduction=self.reduction), self.gamma)


class ScoreBasedConLoss(nn.Module):
    def __init__(self, base_temperature=0.07):
        super(ScoreBasedConLoss, self).__init__()
        self.base_temperature = base_temperature

    def forward(self, features, scores, temp=0.1):
        # features has shape (bs,64)
        # scores has shape (bs)
        # univ (14_20) 0.1, 0.5 >>> All (0.16,0.33) Challenging (0.32,0.69)
        # univ (15_07) 0.1, 0.7 >>> All (0.16,0.33) Challenging (0.32,0.68)
        # univ (15_44) 0.08,0.7 >>> All (0.16,0.33) Challenging (0.32,0.69)
        # univ (16_26) 0.1, 0.5 (200) >>> All (0.16,0.32) Challenging (0.32,0.69)
        # univ (17_39) 0.1, 0.7 (start: epe-10) >>> All (0.16,0.33) Challenging (0.32,0.70)
        # univ (18_02) 0.1, 0.7 (start: epe-5) >>> All (0.16,0.33) Challenging (0.33,0.72)
        # univ (18_24) 0.1, 0.7 (start: epe-2) >>> All (0.16,0.33) Challenging (0.32,0.70)
        # univ (19_) 0.1, 0.7 (no head) >>> All (0.16,0.32) Challenging (0.32,0.69)
        # zara1 () 0.1, 0.7
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        scores = scores.contiguous().view(-1, 1)  # (bs,1)
        mask_positives = (torch.abs(scores.sub(scores.T)) < 0.1).float().to(device)  # (bs,bs)  # 0.1, 0.08
        mask_negatives = (torch.abs(scores.sub(scores.T)) > 1.0).float().to(device)  # (bs,bs)  # 0.5, 0.7
        mask_neutral = mask_positives + mask_negatives  # the zeros elements represent the neutral samples

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)  # (bs,bs)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (bs,1)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask_positives),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        ) * mask_neutral

        mask_positives = mask_positives * logits_mask  # (bs,bs)

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-20)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_positives * log_prob).sum(1) / (mask_positives.sum(1) + 1e-20)

        # loss
        loss = - (temp / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()
