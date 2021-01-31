import warnings

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from environment.scene_graph import DirectedEdge

import model.dynamics as dynamic_module
from model.components import *
from model.model_utils import *


class MultiHypothesisNet(object):
    def __init__(self,
                 env,
                 node_type,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 log_writer=None):
        self.hyperparams = hyperparams
        self.env = env
        self.node_type = node_type
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [
            edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.nb_classes = self.hyperparams['num_classes']
        self.state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims)
                        for entity_dims in self.state[env.robot_type].values()])
            )
        self.pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        edge_types_str = [DirectedEdge.get_str_from_types(
            *edge_type) for edge_type in self.edge_types]
        self.create_graphical_model(edge_types_str)

        dynamic_class = getattr(
            dynamic_module, hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(
            name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=nn.LSTM(input_size=self.state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                                   batch_first=True))

        ############################
        #   Robot Future Encoder   #
        ############################
        # We'll create this here, but then later check if we're next to the robot.
        # Based on that, we'll factor this into the computation graph (or not).
        if self.hyperparams['incl_robot_node']:
            self.add_submodule('robot_future_encoder',
                               model_if_absent=nn.LSTM(input_size=self.robot_state_length,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                       bidirectional=True,
                                                       batch_first=True))
            # These are related to how you initialize states for the robot future encoder.
            self.add_submodule('robot_future_encoder/initial_h',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))
            self.add_submodule('robot_future_encoder/initial_c',
                               model_if_absent=nn.Linear(self.robot_state_length,
                                                         self.hyperparams['enc_rnn_dim_future']))

        if self.hyperparams['edge_encoding']:
            ##############################
            #   Edge Influence Encoder   #
            ##############################
            # NOTE: The edge influence encoding happens during calls
            # to forward or incremental_forward, so we don't create
            # a model for it here for the max and sum variants.
            if self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=nn.LSTM(input_size=self.hyperparams['enc_rnn_dim_edge'],
                                                           hidden_size=self.hyperparams['enc_rnn_dim_edge_influence'],
                                                           bidirectional=True,
                                                           batch_first=True))

                # Four times because we're trying to mimic a bi-directional
                # LSTM's output (which, here, is c and h from both ends).
                self.eie_output_dims = 4 * \
                    self.hyperparams['enc_rnn_dim_edge_influence']

            elif self.hyperparams['edge_influence_combine_method'] == 'attention':
                # Chose additive attention because of https://arxiv.org/pdf/1703.03906.pdf
                # We calculate an attention context vector using the encoded edges as the "encoder"
                # (that we attend _over_)
                # and the node history encoder representation as the "decoder state" (that we attend _on_).
                self.add_submodule(self.node_type + '/edge_influence_encoder',
                                   model_if_absent=AdditiveAttention(
                                       encoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_edge_influence'],
                                       decoder_hidden_state_dim=self.hyperparams['enc_rnn_dim_history']))

                self.eie_output_dims = self.hyperparams['enc_rnn_dim_edge_influence']

        ###################
        #   Map Encoder   #
        ###################
        if self.hyperparams['use_map_encoding']:
            if self.node_type in self.hyperparams['map_encoder']:
                me_params = self.hyperparams['map_encoder'][self.node_type]
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']

        if self.hyperparams['edge_encoding']:
            #              Edge Encoder
            x_size += self.eie_output_dims
        if self.hyperparams['incl_robot_node']:
            #              Future Conditional Encoder
            x_size += 4 * self.hyperparams['enc_rnn_dim_future']
        if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
            #              Map Encoder
            x_size += self.hyperparams['map_encoder'][self.node_type]['output_size']

        ####################
        #   Decoder LSTM   #
        ####################

        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length + self.robot_state_length + x_size
        else:
            decoder_input_dims = self.pred_state_length + x_size

        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(x_size, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_mu',
                           model_if_absent=nn.Linear(self.state_length, self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/kalman_logits',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'] + decoder_input_dims, self.nb_classes))
        self.x_size = x_size

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[edge_type.split('->')[1]].values()]))
            if self.hyperparams['edge_state_combine_method'] == 'pointnet':
                self.add_submodule(edge_type + '/pointnet_encoder',
                                   model_if_absent=nn.Sequential(
                                       nn.Linear(self.state_length,
                                                 2 * self.state_length),
                                       nn.ReLU(),
                                       nn.Linear(2 * self.state_length,
                                                 2 * self.state_length),
                                       nn.ReLU()))

                edge_encoder_input_size = 2 * self.state_length + self.state_length

            elif self.hyperparams['edge_state_combine_method'] == 'attention':
                self.add_submodule(self.node_type + '/edge_attention_combine',
                                   model_if_absent=TemporallyBatchedAdditiveAttention(
                                       encoder_hidden_state_dim=self.state_length,
                                       decoder_hidden_state_dim=self.state_length))
                edge_encoder_input_size = self.state_length + neighbor_state_length

            else:
                edge_encoder_input_size = self.state_length + neighbor_state_length

            self.add_submodule(edge_type + '/edge_encoder',
                               model_if_absent=nn.LSTM(input_size=edge_encoder_input_size,
                                                       hidden_size=self.hyperparams['enc_rnn_dim_edge'],
                                                       batch_first=True))

    def create_graphical_model(self, edge_types):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        #####################
        #   Edge Encoders   #
        #####################
        if self.hyperparams['edge_encoding']:
            self.create_edge_models(edge_types)

        for name, module in self.node_modules.items():
            module.to(self.device)

    def obtain_encoded_tensors(self,
                               mode,
                               inputs,
                               inputs_st,
                               labels,
                               labels_st,
                               first_history_indices,
                               neighbors,
                               neighbors_edge_value,
                               robot,
                               map) -> (torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor,
                                        torch.Tensor):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        # TODO input = Node History [256, 8, 6]
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
                          # TODO List of BS of list of nb neighbors of tensors [t, neighbor state]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        # TODO List of Bs of tensors of shape [nb neighbors]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot. (Future Motion Plan of ego agent)
            - y: Label / future of the node. (Ground truth)
            - n_s_t0: Standardized current state of the node.
        """
        with torch.no_grad():
            x, x_r_t, y_e, y_r, y = None, None, None, None, None
            initial_dynamics = dict()

            batch_size = inputs.shape[0]
            #########################################
            # Provide basic information to encoders #
            #########################################
            node_history = inputs
            node_present_state = inputs[:, -1]
            node_pos = inputs[:, -1, 0:2]
            node_vel = inputs[:, -1, 2:4]

            node_history_st = inputs_st
            node_present_state_st = inputs_st[:, -1]
            node_pos_st = inputs_st[:, -1, 0:2]
            node_vel_st = inputs_st[:, -1, 2:4]

            n_s_t0 = node_present_state_st

            initial_dynamics['pos'] = node_pos
            initial_dynamics['vel'] = node_vel

            self.dynamic.set_initial_condition(initial_dynamics)

            if self.hyperparams['incl_robot_node']:
                x_r_t, y_r = robot[..., 0, :], robot[..., 1:, :]

            ##################
            # Encode History #
            ##################
            node_history_encoded = self.encode_node_history(mode,
                                                            node_history_st,
                                                            first_history_indices)

            ##################
            # Encode Present #
            ##################
            node_present = node_present_state_st  # [bs, state_dim]

            ##################
            # Encode Future #
            ##################
            y = labels_st

            ##############################
            # Encode Node Edges per Type #
            ##############################
            if self.hyperparams['edge_encoding']:
                node_edges_encoded = list()
                for edge_type in self.edge_types:
                    # Encode edges for given edge type
                    encoded_edges_type = self.encode_edge(mode,
                                                          node_history,
                                                          node_history_st,
                                                          edge_type,
                                                          neighbors[edge_type],
                                                          neighbors_edge_value[edge_type],
                                                          first_history_indices)
                    # List of [bs/nbs, enc_rnn_dim]
                    node_edges_encoded.append(encoded_edges_type)
                #####################
                # Encode Node Edges #
                #####################
                total_edge_influence = self.encode_total_edge_influence(mode,
                                                                        node_edges_encoded,
                                                                        node_history_encoded,
                                                                        batch_size)

            ################
            # Map Encoding #
            ################
            if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
                if self.log_writer and (self.curr_iter + 1) % 500 == 0:
                    map_clone = map.clone()
                    map_patch = self.hyperparams['map_encoder'][self.node_type]['patch_size']
                    map_clone[:, :, map_patch[1]-5:map_patch[1] +
                              5, map_patch[0]-5:map_patch[0]+5] = 1.
                    self.log_writer.add_images(f"{self.node_type}/cropped_maps", map_clone,
                                               self.curr_iter, dataformats='NCWH')

                encoded_map = self.node_modules[self.node_type +
                                                '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
                do = self.hyperparams['map_encoder'][self.node_type]['dropout']
                encoded_map = F.dropout(
                    encoded_map, do, training=(mode == ModeKeys.TRAIN))

            ######################################
            # Concatenate Encoder Outputs into x #
            ######################################
            x_concat_list = list()

            # Every node has an edge-influence encoder (which could just be zero).
            if self.hyperparams['edge_encoding']:
                # [bs/nbs, 4*enc_rnn_dim]
                x_concat_list.append(total_edge_influence)

            # Every node has a history encoder.
            # [bs/nbs, enc_rnn_dim_history]
            x_concat_list.append(node_history_encoded)

            if self.hyperparams['incl_robot_node']:
                robot_future_encoder = self.encode_robot_future(mode, x_r_t, y_r)
                x_concat_list.append(robot_future_encoder)

            if self.hyperparams['use_map_encoding'] and self.node_type in self.hyperparams['map_encoder']:
                if self.log_writer:
                    self.log_writer.add_scalar(f"{self.node_type}/encoded_map_max",
                                               torch.max(torch.abs(encoded_map)), self.curr_iter)
                x_concat_list.append(encoded_map)

            x = torch.cat(x_concat_list, dim=1)

        return x, n_s_t0, x_r_t

    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules[self.node_type + '/node_history_encoder'],
                                                      original_seqs=node_hist,
                                                      lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1. -
                            self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)

        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_edge(self,
                    mode,
                    node_history,
                    node_history_st,
                    edge_type,
                    neighbors,
                    neighbors_edge_value,
                    first_history_indices):
        with torch.no_grad():
            max_hl = self.hyperparams['maximum_history_length']

            edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
            # Get neighbors for timestep in batch
            for i, neighbor_states in enumerate(neighbors):
                if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                    neighbor_state_length = int(
                        np.sum([len(entity_dims)
                                for entity_dims in self.state[edge_type[1]].values()])
                    )
                    edge_states_list.append(torch.zeros(
                        (1, max_hl + 1, neighbor_state_length), device=self.device))
                else:
                    edge_states_list.append(torch.stack(
                        neighbor_states, dim=0).to(self.device))
            # TODO This results => list of Bs tensors of shape [7, 8, 6]
            if self.hyperparams['edge_state_combine_method'] == 'sum':
                # Used in Structural-RNN to combine edges as well.
                op_applied_edge_states_list = list()
                for neighbors_state in edge_states_list:
                    op_applied_edge_states_list.append(
                        torch.sum(neighbors_state, dim=0))
                combined_neighbors = torch.stack(
                    op_applied_edge_states_list, dim=0)
                # TODO This results => in tensor [Bs, T, State]
                if self.hyperparams['dynamic_edges'] == 'yes':
                    # Should now be (bs, time, 1)
                    op_applied_edge_mask_list = list()
                    for edge_value in neighbors_edge_value:
                        op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),
                                                                               dim=0, keepdim=True), max=1.))
                    combined_edge_masks = torch.stack(
                        op_applied_edge_mask_list, dim=0)

            elif self.hyperparams['edge_state_combine_method'] == 'max':
                # Used in NLP, e.g. max over word embeddings in a sentence.
                op_applied_edge_states_list = list()
                for neighbors_state in edge_states_list:
                    op_applied_edge_states_list.append(
                        torch.max(neighbors_state, dim=0))
                combined_neighbors = torch.stack(
                    op_applied_edge_states_list, dim=0)
                if self.hyperparams['dynamic_edges'] == 'yes':
                    # Should now be (bs, time, 1)
                    op_applied_edge_mask_list = list()
                    for edge_value in neighbors_edge_value:
                        op_applied_edge_mask_list.append(torch.clamp(torch.max(edge_value.to(self.device),
                                                                               dim=0, keepdim=True), max=1.))
                    combined_edge_masks = torch.stack(
                        op_applied_edge_mask_list, dim=0)

            elif self.hyperparams['edge_state_combine_method'] == 'mean':
                # Used in NLP, e.g. mean over word embeddings in a sentence.
                op_applied_edge_states_list = list()
                for neighbors_state in edge_states_list:
                    op_applied_edge_states_list.append(
                        torch.mean(neighbors_state, dim=0))
                combined_neighbors = torch.stack(
                    op_applied_edge_states_list, dim=0)
                if self.hyperparams['dynamic_edges'] == 'yes':
                    # Should now be (bs, time, 1)
                    op_applied_edge_mask_list = list()
                    for edge_value in neighbors_edge_value:
                        op_applied_edge_mask_list.append(torch.clamp(torch.mean(edge_value.to(self.device),
                                                                                dim=0, keepdim=True), max=1.))
                    combined_edge_masks = torch.stack(
                        op_applied_edge_mask_list, dim=0)

            joint_history = torch.cat(
                [combined_neighbors, node_history_st], dim=-1)
            # TODO => joint history combind neighbors [Bs, T, State] and Ego history with [Bs, T, State] => [Bs, T, State*2]

            outputs, _ = run_lstm_on_variable_length_seqs(
                self.node_modules[DirectedEdge.get_str_from_types(
                    *edge_type) + '/edge_encoder'],
                original_seqs=joint_history,
                lower_indices=first_history_indices
            )

            outputs = F.dropout(outputs,
                                p=1. -
                                self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

            last_index_per_sequence = -(first_history_indices + 1)
            ret = outputs[torch.arange(
                last_index_per_sequence.shape[0]), last_index_per_sequence]
        if self.hyperparams['dynamic_edges'] == 'yes':
            return ret * combined_edge_masks
        else:
            return ret

    def encode_total_edge_influence(self, mode, encoded_edges, node_history_encoder, batch_size):
        if self.hyperparams['edge_influence_combine_method'] == 'sum':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.sum(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'mean':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.mean(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'max':
            stacked_encoded_edges = torch.stack(encoded_edges, dim=0)
            combined_edges = torch.max(stacked_encoded_edges, dim=0)

        elif self.hyperparams['edge_influence_combine_method'] == 'bi-rnn':
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros(
                    (batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                _, state = self.node_modules[self.node_type +
                                             '/edge_influence_encoder'](encoded_edges)
                combined_edges = unpack_RNN_state(state)
                combined_edges = F.dropout(combined_edges,
                                           p=1. -
                                           self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))

        elif self.hyperparams['edge_influence_combine_method'] == 'attention':
            # Used in Social Attention (https://arxiv.org/abs/1710.04689)
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros(
                    (batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)
                combined_edges, _ = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges,
                                                                                                  node_history_encoder)
                combined_edges = F.dropout(combined_edges,
                                           p=1. -
                                           self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))
        return combined_edges

    def encode_robot_future(self, mode, robot_present, robot_future) -> torch.Tensor:
        """
        Encodes the robot future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param robot_present: Current state of the robot. [bs, state]
        :param robot_future: Future states of the robot. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules['robot_future_encoder/initial_h']
        initial_c_model = self.node_modules['robot_future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(robot_present)
        initial_h = torch.stack(
            [initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(robot_present)
        initial_c = torch.stack(
            [initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules['robot_future_encoder'](
            robot_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. -
                          self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state

    def decoder(self, x, n_s_t0, x_nr_t):
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']
        initial_mu_model = self.node_modules[self.node_type + '/decoder/initial_mu']
        logits_model = self.node_modules[self.node_type + '/decoder/kalman_logits']
        initial_h = initial_h_model(x)
        initial_mu = initial_mu_model(n_s_t0)

        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([x, initial_mu, x_nr_t], dim=1)
        else:
            input_ = torch.cat([x, initial_mu], dim=1)

        features = torch.cat([input_, initial_h], dim=1)
        logits = logits_model(features)
        return logits, features

    def predict_kalman_class(self, x, n_s_t0, x_nr_t):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return:
        """
        y_predicted, features = self.decoder(x, n_s_t0, x_nr_t)
        return y_predicted, features
