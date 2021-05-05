import numpy as np
import torch

from model.dataset import get_timesteps_data, restore
from model.mgcvae import MultimodalGenerativeCVAE
from model.multi_hyp_groupExperts import MultiHypothesisNet


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device, class_count_dict= None, train_borders = None, train_borders_match_class_per_bin = None, joint_train= True):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0
        self.class_count_dict = class_count_dict
        self.train_borders = train_borders
        self.train_borders_match_class_per_bin = train_borders_match_class_per_bin

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        self.joint_train = joint_train
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims)
                        for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultiHypothesisNet(env,
                                                                      node_type,
                                                                      self.model_registrar,
                                                                      self.hyperparams,
                                                                      self.device,
                                                                      edge_types,
                                                                      train_borders = self.train_borders,
                                                                      train_borders_match_class_per_bin = self.train_borders_match_class_per_bin,
                                                                      class_count_dict = self.class_count_dict,
                                                                      log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

#     def set_annealing_params(self):
#         for node_str, model in self.node_models_dict.items():
#             model.set_annealing_params()

#     def step_annealers(self, node_type=None):
#         if node_type is None:
#             for node_type in self.node_models_dict:
#                 self.node_models_dict[node_type].step_annealers()
#         else:
#             self.node_models_dict[node_type].step_annealers()

    def train_loss(self, batch, node_type, weight, top_n=8):
        # TODO DATA explained here
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map, target_class, target_class_bin, scores) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        target_class = target_class.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        if self.joint_train:
            loss_reg, loss_cl, loss_cl_no_reweighted, logits = model.train_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    top_n=top_n, 
                                    targets_cl = target_class,
                                    weight = weight,
                                    joint_train = self.joint_train)

            return loss_reg, loss_cl, loss_cl_no_reweighted.detach(), logits
        else:
            loss, loss_no_reweighted, logits = model.train_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    top_n=top_n, 
                                    targets_cl = target_class,
                                    weight = weight,
                                    joint_train = self.joint_train)

            return loss, loss_no_reweighted.detach(), logits
    

    def train_lossGroupExperts(self, batch, node_type, weight, nb_bins, top_n=8):
        # TODO DATA explained here
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map, target_class, target_class_bin, scores) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        target_class = target_class.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        if self.joint_train:
            loss_reg, loss_cl,loss_cl_no_reweighted, accuracies_bins = model.train_lossGroupExperts(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    top_n=top_n, 
                                    targets_cl = target_class,
                                    nb_bins = nb_bins, 
                                    target_class_bin = target_class_bin, 
                                    weight = weight,
                                    joint_train = self.joint_train)
            return loss_reg, loss_cl,loss_cl_no_reweighted, accuracies_bins
        else:
            loss, loss_no_reweighted, logits = model.train_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    top_n=top_n, 
                                    targets_cl = target_class,
                                    weight = weight,
                                    joint_train = self.joint_train)

            return loss, loss_no_reweighted.detach(), logits

    def train_lossGroupExperts_con(self, batch, node_type, top_n=8):
        # TODO DATA explained here
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map, target_class, target_class_bin, scores) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        target_class = target_class.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        if self.joint_train:
            loss_reg, loss_cl = model.train_lossGroupExperts_con(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    top_n=top_n, 
                                    scores = scores,
                                    joint_train = self.joint_train)

            return loss_reg, loss_cl
        else:
            loss = model.train_lossGroupExperts_con(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    top_n=top_n, 
                                    scores = scores,
                                    joint_train = self.joint_train)

            return loss

    def eval_loss(self, batch, node_type, weight):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map, target_class) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        target_class = target_class.to(self.device)
        # Run forward pass
        model = self.node_models_dict[node_type]
        if self.joint_train:
            loss_reg, loss_cl, loss_cl_no_reweighted, logits = model.eval_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    targets_cl = target_class,
                                    weight = weight,
                                    joint_train = self.joint_train)

            return loss_reg, loss_cl, loss_cl_no_reweighted.detach(), logits
        else:
            nll = model.eval_loss(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                neighbors=restore(neighbors_data_st),
                                neighbors_edge_value=restore(
                                    neighbors_edge_value),
                                robot=robot_traj_st_t,
                                map=map,
                                prediction_horizon=self.ph)

            return nll.cpu().detach().numpy()

    def eval_lossGroupExperts(self, batch, node_type, weight, nb_bins):
        # TODO DATA explained here
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map, target_class, target_class_bin, scores) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        target_class = target_class.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        if self.joint_train:
            loss_reg, loss_cl,loss_cl_no_reweighted, accuracies_bins = model.eval_lossGroupExperts(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    targets_cl = target_class,
                                    nb_bins = nb_bins, 
                                    target_class_bin = target_class_bin, 
                                    weight = weight,
                                    joint_train = self.joint_train)
            return loss_reg, loss_cl,loss_cl_no_reweighted, accuracies_bins
        else:
            loss, loss_no_reweighted, logits = model.eval_loss(inputs=x,
                                    inputs_st=x_st_t,
                                    first_history_indices=first_history_index,
                                    labels=y,
                                    labels_st=y_st_t,
                                    neighbors=restore(neighbors_data_st),
                                    neighbors_edge_value=restore(
                                        neighbors_edge_value),
                                    robot=robot_traj_st_t,
                                    map=map,
                                    prediction_horizon=self.ph,
                                    targets_cl = target_class,
                                    weight = weight,
                                    joint_train = self.joint_train)

            return loss, loss_no_reweighted.detach(), logits


    def predict(self,
                scene,
                timesteps,
                ph,
                joint_train, 
                num_samples=1,
                min_future_timesteps=0,
                min_history_timesteps=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):
        predictions_dict = {}
        features_list = []
        predictions_cl_list = []
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            (first_history_index,
             x_t, y_t, x_st_t, y_st_t,
             neighbors_data_st,
             neighbors_edge_value,
             robot_traj_st_t,
             map), nodes, timesteps_o = batch

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.Tensor:
                map = map.to(self.device)

            # Run forward pass
            predictions, features, predictions_cl = model.predict(inputs=x,
                                        inputs_st=x_st_t,
                                        first_history_indices=first_history_index,
                                        neighbors=neighbors_data_st,
                                        neighbors_edge_value=neighbors_edge_value,
                                        robot=robot_traj_st_t,
                                        map=map,
                                        prediction_horizon=ph,
                                        num_samples=num_samples,
                                        joint_train = joint_train,
                                        z_mode=z_mode,
                                        gmm_mode=gmm_mode,
                                        full_dist=full_dist,
                                        all_z_sep=all_z_sep)

            features_list.append(features)
            predictions_cl_list = predictions_cl.tolist()

            predictions_np = predictions.cpu().detach().numpy() #[bs, num_hyp, horizon, 2]
            # predictions in trajectron  should be: [num_hyp, bs, horizon, 2]
            predictions_np = np.transpose(predictions_np, (1,0,2,3))

            # Assign predictions to node
            for i, ts in enumerate(timesteps_o):
                if ts not in predictions_dict.keys():
                    predictions_dict[ts] = dict()
                predictions_dict[ts][nodes[i]] = np.transpose(
                    predictions_np[:, [i]], (1, 0, 2, 3))

        return predictions_dict, features_list, predictions_cl_list
