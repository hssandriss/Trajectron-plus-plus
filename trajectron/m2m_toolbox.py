import warnings
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from torch import Size, nn, tensor
from tqdm import tqdm

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


def train_epoch(trajectron, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                train_data_loader, epoch, hyperparams, log_writer, device):

    trajectron.model_registrar.train()

    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        loss_epoch = []
        correct_epoch = 0
        class_loss = {k: [] for k in hyperparams['class_count_dic'].keys()}
        class_correct = {k: 0 for k in hyperparams['class_count_dic'].keys()}
        class_count_sampled = {k: 0 for k in hyperparams['class_count_dic'].keys()}
        log_writer.add_scalar(f"{node_type}/classification_g/train/lr_scheduling",
                              lr_scheduler[node_type].state_dict()['_last_lr'][0], epoch)
        pbar = tqdm(data_loader, ncols=120)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            optimizer[node_type].zero_grad()
            inputs = batch[:-1]
            targets = batch[-1]
            targets = targets.to(device)
            # inputs_shape =[x[i].shape for i in range(5)]
            inputs = trajectron.preprocess_edges(inputs, node_type)
            y_hat, features = trajectron.predict(inputs, node_type)
            train_loss = criterion(y_hat, targets)
            # import pdb; pdb.set_trace()
            pbar.set_description(
                f"Epoch {epoch}, {node_type} L: {train_loss.mean().item():.2f} (log10: {train_loss.mean().log10().item():.2f})")
            train_loss.mean().backward()
            # Clipping gradients.
            if hyperparams['grad_clip'] is not None:
                nn.utils.clip_grad_value_(trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
            optimizer[node_type].step()
            # Stepping forward the learning rate scheduler and annealers.
            # Per class metrics
            predicted = torch.argmax(F.softmax(y_hat, 1), 1)
            loss_epoch.append(train_loss.clone().detach())  # [[bs, 1], [bs, 1]...]
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
        loss = torch.cat(loss_epoch)
        assert class_count_sampled == hyperparams['class_count_dic'], "You didn't go through all data"
        log_writer.add_scalar(f"{node_type}/classification_g/train/loss", loss.mean().log10().item(), epoch)
        log_writer.add_scalar(f"{node_type}/classification_g/train/loss_logarithmic", loss.mean().item(), epoch)
        log_writer.add_scalar(f"{node_type}/classification_g/train/accuracy",
                              correct_epoch / data_loader.dataset.len, epoch)

        ret_class_acc = {k: class_correct[k] / hyperparams['class_count_dic'][k]
                         for k in hyperparams['class_count_dic'].keys()}
        ret_class_loss_log = {k: torch.cat(class_loss[k]).mean().log10().item()
                              for k in hyperparams['class_count_dic'].keys()}
        ret_class_loss = {k: torch.cat(class_loss[k]).mean().item()
                          for k in hyperparams['class_count_dic'].keys()}
        for k in hyperparams['class_count_dic'].keys():
            log_writer.add_scalar(f"{node_type}/classification_g/train/loss_class_{k}", ret_class_loss[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification_g/train/loss_logarithmic_{k}",
                                  ret_class_loss_log[k], epoch)
            log_writer.add_scalar(f"{node_type}/classification_g/train/accuracy_class_{k}", ret_class_acc[k], epoch)

        print("Epoch Loss: " + bcolors.OKGREEN + str(round(loss.mean().log10().item(), 3)) + bcolors.ENDC)
        print("Epoch Accuracy: " + bcolors.OKGREEN + str(round(correct_epoch / data_loader.dataset.len, 3)) + bcolors.ENDC)
        print("Accuracy per class: ")
        print(bcolors.OKGREEN + str({k: round(ret_class_acc[k], 3)
                                     for k in hyperparams['class_count_dic'].keys()}) + bcolors.ENDC)
    return ret_class_acc, ret_class_loss


def train_epoch_con(trajectron, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                    train_data_loader, epoch, hyperparams, log_writer, device):

    trajectron.model_registrar.train()

    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        loss_epoch = []
        pos_epoch = []
        neg_epoch = []
        log_writer.add_scalar(f"{node_type}/classification_g/train/lr_scheduling",
                              lr_scheduler[node_type].state_dict()['_last_lr'][0], epoch)
        pbar = tqdm(data_loader, ncols=120)
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            optimizer[node_type].zero_grad()
            inputs = batch[:-1]
            targets = batch[-1]
            targets = targets.to(device)
            inputs = trajectron.preprocess_edges(inputs, node_type)
            _, features = trajectron.predict(inputs, node_type)
            train_loss, mask_pos, mask_neg = criterion(features, targets)
            pbar.set_description(
                f"Epoch {epoch}, {node_type} L: {train_loss.item():.2f} Positives: {mask_pos.item():.2f}: Negatives: {mask_neg.item():.2f} ")
            train_loss.register_hook(lambda grad: print(grad))
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
        log_writer.add_scalar(f"{node_type}/classification_g/train/conloss", np.mean(loss_epoch), epoch)
        log_writer.add_scalar(f"{node_type}/classification_g/train/avg_positive_samples", np.mean(pos_epoch), epoch)
        log_writer.add_scalar(f"{node_type}/classification_g/train/avg_negative_samples", np.mean(neg_epoch), epoch)
        print("Epoch Loss: " + bcolors.OKGREEN + str(round(np.mean(loss_epoch), 3)) + bcolors.ENDC)
    return np.mean(loss_epoch)


def classwise_loss(outputs, targets):
    """
     Returns logit confidence
    """
    out_1hot = torch.zeros_like(outputs)
    out_1hot.scatter_(1, targets.view(-1, 1), 1)
    return (outputs * out_1hot).sum(1).mean()


def make_step(grad, attack, step_size):
    if attack == 'l2':
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1, 1)
        scaled_grad = grad / (grad_norm + 1e-10)
        step = step_size * scaled_grad
    elif attack == 'inf':
        step = step_size * torch.sign(grad)
    else:
        step = step_size * grad
    return step


def random_perturb(inputs, attack, eps):
    if attack == 'inf':
        r_inputs = 2 * (torch.rand_like(inputs) - 0.5) * eps
    else:
        r_inputs = (torch.rand_like(inputs) - 0.5).renorm(p=2, dim=1, maxnorm=eps)
    return r_inputs


def sum_tensor(tensor):
    return tensor.float().sum().item()


def generation(trajectron_g, trajectron, node_type, device, seed_inputs, seed_targets, gen_targets, p_accept,
               gamma, lam, step_size, random_start=True, max_iter=10):
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
    trajectron_g.model_registrar.eval()
    trajectron.model_registrar.eval()
    criterion = nn.CrossEntropyLoss()
    (first_history_index, x_t, y_t, x_st_t, y_st_t, preprocessed_edges, robot_traj_st_t, map) = seed_inputs
    # We will use x_st_t and preprocessed_edges
    # Random Noise
    if random_start:
        # add random noise to node states
        random_noise = random_perturb(x_st_t, 'l2', 0.5)
        import pdb; pdb.set_trace()
        x_st_t = torch.clamp(x_st_t + random_noise, 0, 1)
        # add random noise to edge data
        for key in preprocessed_edges.keys():
            random_noise = random_perturb(preprocessed_edges[key][0], 'l2', 0.5)
            import pdb; pdb.set_trace()
            preprocessed_edges[key][0] = torch.clamp(preprocessed_edges[key][0] + random_noise, 0, 1)
            random_noise = random_perturb(preprocessed_edges[key][1], 'l2', 0.5)
            import pdb; pdb.set_trace()
            preprocessed_edges[key][1] = torch.clamp(preprocessed_edges[key][1] + random_noise, 0, 1)
    else:
        inputs_ = seed_inputs
    # Verify the shapes
    assert len(inputs_[0].shape) == 2
    assert inputs_[0].shape[0] == seed_targets.shape[0]
    assert inputs_[0].shape[1] == 64
    # Loop for optimizing the objective
    for _ in range(max_iter):
        inputs_ = [tensor.clone().detach().requires_grad_(True) if tensor is not None else None
                   for tensor in inputs_]
        outputs_g, _ = trajectron_g.predict_kalman_class(*inputs_, node_type)
        outputs_r, _ = trajectron.predict_kalman_class(*inputs_, node_type)

        loss = criterion(outputs_g, gen_targets) + lam * classwise_loss(outputs_r, seed_targets)
        # Calculate grad with respect to each part of the input: x, n_s_t0, x_nr_t
        if trajectron.hyperparams['incl_robot_node']:
            x, n_s_t0, x_nr_t = inputs_
            grad_x, grad_x_s_t0, grad_x_nr_t = torch.autograd.grad(loss, [x, n_s_t0, x_nr_t])
            x = x - make_step(grad_x, 'l2', step_size)
            n_s_t0 = n_s_t0 - make_step(grad_x_s_t0, 'l2', step_size)
            x_nr_t = x_nr_t - make_step(grad_x_nr_t, 'l2', step_size)
            inputs_ = [torch.clamp(x, 0, 1), torch.clamp(n_s_t0, 0, 1), torch.clamp(x_nr_t, 0, 1)]
        else:
            x, n_s_t0, _ = inputs_
            grad_x, grad_x_s_t0 = torch.autograd.grad(loss, [x, n_s_t0])
            x = x - make_step(grad_x, 'l2', step_size)
            n_s_t0 = n_s_t0 - make_step(grad_x_s_t0, 'l2', step_size)
            inputs_ = [torch.clamp(x, 0, 1), torch.clamp(n_s_t0, 0, 1), None]
    # inputs_ = inputs_.detach()
    inputs_ = [tensor.detach() if tensor is not None else None for tensor in inputs_]
    outputs_g, _ = trajectron_g.predict_kalman_class(*inputs_, node_type)
    # seed_targets should be shifted to -> targets
    # To check the current outputs targets:
    predicted_classes = torch.argmax(F.softmax(outputs_g, 1), dim=1)
    meta_count = (gen_targets == predicted_classes).sum().item()
    # import pdb; pdb.set_trace()
    # one_hot is the expected output if we generated the goal targets k
    one_hot = torch.zeros_like(outputs_g)
    one_hot.scatter_(1, gen_targets.view(-1, 1), 1)
    # probs_g is the probabilites of k* in output_g
    probs_g = torch.softmax(outputs_g, dim=1)[one_hot.to(torch.bool)]
    # correct is the condition that indicates if x* can be used as part of samples from k*
    correct = ((probs_g >= gamma) * torch.bernoulli(p_accept)).type(torch.bool).to(device)

    trajectron.model_registrar.train()
    return inputs_, correct, meta_count

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


def select_from_batch_input(input, select_idx):
    ret = []
    for e in input:
        if isinstance(e, torch.Tensor):
            ret.append(e[select_idx])
        elif isinstance(e, dict) and isinstance(list(e.values())[0], list):
            dic = {}
            for k, v in e.items():
                assert all(isinstance(elem, torch.Tensor) for elem in v)
                dic[k] = [tensor[select_idx] for tensor in v]
            ret.append(dic)
        else:
            ret.append(None)
    return ret


def train_net(trajectron, trajectron_g, node_type, criterion, optimizer, lr_scheduler, inputs_orig_tuple,
              targets_orig, gen_idx, gen_targets, hyperparams, device):
    class_gen_batch = {k: 0 for k in hyperparams['class_count_dic'].keys()}
    class_loss_batch = {k: 0 for k in hyperparams['class_count_dic'].keys()}
    class_acc_batch = {k: 0 for k in hyperparams['class_count_dic'].keys()}
    ########################
    batch_size = inputs_orig_tuple[0].size(0)
    ########################
    # inputs = inputs_orig_tuple.clone()

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
    # ! Now we have sampled seed classes k0 of initial point x0 given gen_target class k.
    gen_inputs, correct_mask, m_count = generation(trajectron_g, trajectron, node_type, device, seed_inputs, seed_targets, gen_targets,
                                                   p_accept, hyperparams['gamma'], hyperparams['lam'], hyperparams['step_size'], True, hyperparams['attack_iter'])
    #######################
    # Only change the correctly generated samples
    num_gen = sum_tensor(correct_mask)
    num_others = batch_size - num_gen
    gen_c_idx = gen_idx[correct_mask]
    others_mask = torch.ones(batch_size, dtype=torch.bool, device=device)
    others_mask[gen_c_idx] = 0
    others_idx = others_mask.nonzero().view(-1)
    x, n_s_t0, x_nr_t = inputs_tuple
    if num_gen > 0:
        gen_inputs_c = tuple(
            tensor[correct_mask] if tensor is not None else None for tensor in gen_inputs)
        gen_x_c, gen_n_s_t0_c, gen_x_nr_t_c = gen_inputs_c
        gen_targets_c = gen_targets[correct_mask]

        targets[gen_c_idx] = gen_targets_c
        x[gen_c_idx] = gen_x_c
        n_s_t0[gen_c_idx] = gen_n_s_t0_c
        if trajectron.hyperparams['incl_robot_node']:
            x_nr_t[gen_c_idx] = gen_x_nr_t_c
        x_orig, n_s_t0_orig, _ = inputs_orig_tuple
        assert (targets != targets_orig).sum() == 0
        assert (x != x_orig).sum() > 0 or (n_s_t0 != n_s_t0_orig).sum() > 0

    # ! Bare in mind that here we do not replace the seed targets
    # ! but create new variants of gen_targets[correct_mask]
    # Normal training for a minibatch
    optimizer[node_type].zero_grad()
    y_hat, features = trajectron.predict_kalman_class(x, n_s_t0, x_nr_t, node_type)
    train_loss = criterion(y_hat, targets)
    train_loss.mean().backward()
    # Clipping gradients.
    if hyperparams['grad_clip'] is not None:
        nn.utils.clip_grad_value_(trajectron.model_registrar.parameters(), hyperparams['grad_clip'])
    optimizer[node_type].step()
    # Stepping forward the learning rate scheduler and annealers.
    lr_scheduler[node_type].step()

    ################################
    # Summing up the class based gens, loss, acc for current batch
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

    return oth_loss_total, gen_loss_total, num_others, num_correct_oth, num_gen, num_correct_gen, p_g_orig, p_g_targ, success, targets, m_count, class_gen_batch, class_loss_batch, class_acc_batch


def train_gen_epoch(trajectron, trajectron_g, epoch, curr_iter_node_type, optimizer, lr_scheduler, criterion,
                    train_data_loader, hyperparams, log_writer, device):

    N_SAMPLES_PER_CLASS_T = torch.Tensor(hyperparams['class_count']).to(device)
    trajectron.model_registrar.train()
    trajectron_g.model_registrar.eval()

    results = dict()

    for node_type, data_loader in train_data_loader.items():
        curr_iter = curr_iter_node_type[node_type]
        pbar = tqdm(data_loader, ncols=120)
        oth_loss, gen_loss = 0, 0
        correct_oth = 0
        correct_gen = 0
        total_oth, total_gen = 1e-6, 1e-6
        p_g_orig, p_g_targ = 0, 0
        t_success = torch.zeros(hyperparams['num_classes'], 2)
        class_gen = {k: [] for k in hyperparams['class_count_dic'].keys()}
        class_loss = {k: [] for k in hyperparams['class_count_dic'].keys()}
        class_acc = {k: [] for k in hyperparams['class_count_dic'].keys()}
        for batch in pbar:
            trajectron.set_curr_iter(curr_iter)
            inputs = batch[:-1]
            targets = batch[-1]
            targets = targets.to(device)
            inputs = trajectron.preprocess_edges(inputs, node_type)
            import pdb; pdb.set_trace()
            # Set a generation target for current batch with re-sampling
            # Keep the sample with this probability
            gen_probs = N_SAMPLES_PER_CLASS_T[targets] / N_SAMPLES_PER_CLASS_T[0]
            # Here choose randomly possible gen index according to prob gen_prob choose 1 and 0
            gen_index = (1 - torch.bernoulli(gen_probs)).nonzero()
            gen_index = gen_index.view(-1)
            gen_targets = targets[gen_index]
            t_loss, g_loss, num_others, num_correct, num_gen, num_gen_correct, p_g_orig_batch, p_g_targ_batch, success, final_targets, m_count, class_gen_batch, class_loss_batch, class_acc_batch = train_net(
                trajectron, trajectron_g, node_type, criterion, optimizer, lr_scheduler, inputs, targets, gen_index, gen_targets, hyperparams, device)
            # Count for the modified batch
            pbar.set_description(
                f"Epoch {epoch}, {node_type} #gen_correct: {int(num_gen_correct)} #m_count: {m_count} ")
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
            curr_iter += 1

        results[node_type] = {
            'train_loss': oth_loss / total_oth,
            'gen_loss': gen_loss / total_gen,
            'train_acc': 100. * correct_oth / total_oth,
            'gen_acc': 100. * correct_gen / total_gen,
            'p_g_orig': p_g_orig / total_gen,
            'p_g_targ': p_g_targ / total_gen,
            't_success': t_success
        }
        log_writer.add_scalar(f"{node_type}/classification_f/train/train_loss", oth_loss / total_oth, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/gen_loss", gen_loss / total_gen, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/train_acc", 100. * correct_oth / total_oth, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/gen_acc", 100. * correct_gen / total_gen, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/p_g_orig", p_g_orig / total_gen, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/p_g_targ", p_g_targ / total_gen, epoch)
        log_writer.add_scalar(f"{node_type}/classification_f/train/t_success", t_success, epoch)

        msg = '%s | t_Loss: %.3f | g_Loss: %.3f | Acc: %.3f%% (%d/%d) | Acc_gen: %.3f%% (%d/%d) ' \
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
        print(bcolors.UNDERLINE + "Class Gens:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(ret_class_gen) + bcolors.ENDC)
        print(bcolors.UNDERLINE + "Class Loss:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(ret_class_loss) + bcolors.ENDC)
        print(bcolors.UNDERLINE + "Class Acc:" + bcolors.ENDC)
        print(bcolors.OKBLUE + str(ret_class_acc) + bcolors.ENDC)
        print()

    return results, ret_class_acc, ret_class_loss, ret_class_gen


class LDAMLoss(nn.Module):
    """Reference: https://github.com/kaidic/LDAM-DRW/blob/master/losses.py"""

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, reduction='mean'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.scale = s
        self.weight = weight
        self.reduction = reduction

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.scale * output, target, weight=self.weight, reduction=self.reduction)


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
        # import pdb; pdb.set_trace()
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
