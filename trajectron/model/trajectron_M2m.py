import numpy as np
import torch

from model.dataset import get_timesteps_data, restore
from model.model_utils import *
from model.multi_hyp_M2m import MultiHypothesisNet


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()
        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims)
                        for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']
        self.class_count_dic = self.hyperparams['class_count_dic']

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
                                                                      log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        for node_str, model in self.node_models_dict.items():
            model.set_curr_iter(curr_iter)

    def encoded_x(self, batch, node_type):
        """
        Returns:
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot. (Future Motion Plan of ego agent)
            - y: Label / future of the node. (Ground truth)
            - n_s_t0: Standardized current state of the node.
        """""
        mode = ModeKeys.TRAIN
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t,
         neighbors_data_st,
         neighbors_edge_value,
         robot_traj_st_t,
         map) = batch
        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)
        model = self.node_models_dict[node_type]
        x, n_s_t0, x_nr_t = model.obtain_encoded_tensors(mode=mode,
                                                         inputs=x,
                                                         inputs_st=x_st_t,
                                                         labels=y,
                                                         labels_st=y_st_t,
                                                         first_history_indices=first_history_index,
                                                         neighbors=restore(
                                                             neighbors_data_st),
                                                         neighbors_edge_value=restore(
                                                             neighbors_edge_value),
                                                         robot=robot_traj_st_t,
                                                         map=map)
        return (x, n_s_t0, x_nr_t)

    def predict_kalman_class(self, x, n_s_t0, x_nr_t, node_type):
        model = self.node_models_dict[node_type]
        logits, features = model.predict_kalman_class(x, n_s_t0, x_nr_t)
        return logits, features
