import warnings
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.components import *
from model.model_utils import *
import model.dynamics as dynamic_module
from environment.scene_graph import DirectedEdge


class MultimodalGenerativeCVAE(object):
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
        self.edge_types = [edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[env.robot_type].values()])
            )
        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        edge_types_str = [DirectedEdge.get_str_from_types(*edge_type) for edge_type in self.edge_types]
        self.create_graphical_model(edge_types_str)

        dynamic_class = getattr(dynamic_module, hyperparams['dynamic'][self.node_type]['name'])
        dyn_limits = hyperparams['dynamic'][self.node_type]['limits']
        self.dynamic = dynamic_class(self.env.scenes[0].dt, dyn_limits, device,
                                     self.model_registrar, self.x_size, self.node_type)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

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
                self.eie_output_dims = 4 * self.hyperparams['enc_rnn_dim_edge_influence']

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

        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']
        if self.hyperparams['edge_encoding']:
            #              Edge Encoder
            x_size += self.eie_output_dims

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/p_z_x',
                               model_if_absent=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule(self.node_type + '/hx_to_z',
                           model_if_absent=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule(self.node_type + '/q_z_xy',
                               #                                           Node Future Encoder
                               model_if_absent=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule(self.node_type + '/hxy_to_z',
                           model_if_absent=nn.Linear(hxy_size, self.latent.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        decoder_input_dims = self.pred_state_length * 20 + x_size

        self.add_submodule(self.node_type + '/decoder/state_action',
                           model_if_absent=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))

        self.add_submodule(self.node_type + '/decoder/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(x_size, self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule(self.node_type + '/decoder/proj_to_GMM_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     20 * self.pred_state_length))

        self.x_size = x_size
        self.z_size = z_size

        ## Add the head for the contrastive loss
        self.add_submodule(self.node_type + '/con_head',
                           model_if_absent=nn.Linear(232, 232))

        self.add_submodule(self.node_type + '/fitting_0',
                           model_if_absent=nn.Linear(20 * self.pred_state_length * self.ph, 512))
        self.add_submodule(self.node_type + '/fitting_1',
                           model_if_absent=nn.Linear(512, 20 * 4))

    def create_edge_models(self, edge_types):
        for edge_type in edge_types:
            neighbor_state_length = int(
                np.sum([len(entity_dims) for entity_dims in self.state[edge_type.split('->')[1]].values()]))
            if self.hyperparams['edge_state_combine_method'] == 'pointnet':
                self.add_submodule(edge_type + '/pointnet_encoder',
                                   model_if_absent=nn.Sequential(
                                       nn.Linear(self.state_length, 2 * self.state_length),
                                       nn.ReLU(),
                                       nn.Linear(2 * self.state_length, 2 * self.state_length),
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

    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
            rsetattr(self, name + '_optimizer', dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer,
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['kl_weight_start'],
                                      'finish': self.hyperparams['kl_weight'],
                                      'center_step': self.hyperparams['kl_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                          'kl_sigmoid_divisor']
                                  })

        self.create_new_scheduler(name='latent.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

        #self.summarize_annealers()

    def summarize_annealers(self):
        if self.log_writer is not None:
            for annealed_var in self.annealed_vars:
                if rgetattr(self, annealed_var) is not None:
                    self.log_writer.add_scalar('%s/%s' % (str(self.node_type), annealed_var.replace('.', '/')),
                                               rgetattr(self, annealed_var), self.curr_iter)

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
                                          torch.Tensor):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        x = None
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

        ##################
        # Encode History #
        ##################
        node_history_encoded = self.encode_node_history(mode,
                                                        node_history_st,
                                                        first_history_indices)


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
                node_edges_encoded.append(encoded_edges_type)  # List of [bs/nbs, enc_rnn_dim]
            #####################
            # Encode Node Edges #
            #####################
            total_edge_influence = self.encode_total_edge_influence(mode,
                                                                    node_edges_encoded,
                                                                    node_history_encoded,
                                                                    batch_size)

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()

        # Every node has an edge-influence encoder (which could just be zero).
        if self.hyperparams['edge_encoding']:
            x_concat_list.append(total_edge_influence)  # [bs/nbs, 4*enc_rnn_dim]

        # Every node has a history encoder.
        x_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        x = torch.cat(x_concat_list, dim=1)

        return x, n_s_t0

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
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
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

        max_hl = self.hyperparams['maximum_history_length']

        edge_states_list = list()  # list of [#of neighbors, max_ht, state_dim]
        for i, neighbor_states in enumerate(neighbors):  # Get neighbors for timestep in batch
            if len(neighbor_states) == 0:  # There are no neighbors for edge type # TODO necessary?
                neighbor_state_length = int(
                    np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
                )
                edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
            else:
                edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))

        if self.hyperparams['edge_state_combine_method'] == 'sum':
            # Used in Structural-RNN to combine edges as well.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.sum(edge_value.to(self.device),
                                                                           dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'max':
            # Used in NLP, e.g. max over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.max(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.max(edge_value.to(self.device),
                                                                           dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        elif self.hyperparams['edge_state_combine_method'] == 'mean':
            # Used in NLP, e.g. mean over word embeddings in a sentence.
            op_applied_edge_states_list = list()
            for neighbors_state in edge_states_list:
                op_applied_edge_states_list.append(torch.mean(neighbors_state, dim=0))
            combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0)
            if self.hyperparams['dynamic_edges'] == 'yes':
                # Should now be (bs, time, 1)
                op_applied_edge_mask_list = list()
                for edge_value in neighbors_edge_value:
                    op_applied_edge_mask_list.append(torch.clamp(torch.mean(edge_value.to(self.device),
                                                                            dim=0, keepdim=True), max=1.))
                combined_edge_masks = torch.stack(op_applied_edge_mask_list, dim=0)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.node_modules[DirectedEdge.get_str_from_types(*edge_type) + '/edge_encoder'],
            original_seqs=joint_history,
            lower_indices=first_history_indices
        )

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)
        ret = outputs[torch.arange(last_index_per_sequence.shape[0]), last_index_per_sequence]
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
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)

                _, state = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges)
                combined_edges = unpack_RNN_state(state)
                combined_edges = F.dropout(combined_edges,
                                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))

        elif self.hyperparams['edge_influence_combine_method'] == 'attention':
            # Used in Social Attention (https://arxiv.org/abs/1710.04689)
            if len(encoded_edges) == 0:
                combined_edges = torch.zeros((batch_size, self.eie_output_dims), device=self.device)

            else:
                # axis=1 because then we get size [batch_size, max_time, depth]
                encoded_edges = torch.stack(encoded_edges, dim=1)
                combined_edges, _ = self.node_modules[self.node_type + '/edge_influence_encoder'](encoded_edges,
                                                                                                  node_history_encoder)
                combined_edges = F.dropout(combined_edges,
                                           p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                                           training=(mode == ModeKeys.TRAIN))

        return combined_edges

    def project_to_GMM_params(self, tensor) -> (torch.Tensor):
        mus = self.node_modules[self.node_type + '/decoder/proj_to_GMM_mus'](tensor)
        return mus

    def p_y_xz(self, x, n_s_t0, prediction_horizon):
        ph = prediction_horizon  # 12
        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']

        initial_state = initial_h_model(x)
        mus = []

        # Infer initial action state for node from current state
        a_0 = self.node_modules[self.node_type + '/decoder/state_action'](n_s_t0)

        state = initial_state
        input_ = torch.cat([x, a_0.repeat(1, 20)], dim=1)
        features = torch.cat([input_, state], dim=1)  # bs, 232
        for j in range(ph):
            h_state = cell(input_, state)
            mu_t = self.project_to_GMM_params(h_state)  # bs, 20*2

            mus.append(mu_t.reshape(-1, 20, 2))
            dec_inputs = [x, mu_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        mus = torch.stack(mus, dim=2)  # bs, 20, 12, 2
        y = self.dynamic.integrate_samples(mus, x)  # bs, 20, 12, 2
        return y, features

    def decoder(self, x, n_s_t0, prediction_horizon):
        y, features = self.p_y_xz(x, n_s_t0, prediction_horizon)
        return y, features

    def average_weighted_norm(self, x, w):
        # x shape (bs, 20, 12)
        # w shape (bs, 20)
        w_repeated = w.unsqueeze(-1).repeat(1, 1, 12)  # (bs, 20, 12)
        sum_w = torch.sum(w_repeated, dim=1)  # (bs, 12)
        x_weighted = x * w_repeated  # (bs, 20, 12)
        x_weighted_sum = torch.sum(x_weighted, dim=1)  # (bs, 12)
        result = x_weighted_sum / (sum_w + 1e-6 / 2.0)  # (bs, 12)
        return result

    def get_gaussian_mixture_model_from_samples(self, samples, assignments):
        # samples has shape (bs, 20, 12, 2)
        # assignments has shape (bs, 20, 4)

        num_of_modes = assignments.shape[2]
        assignments_adjusted = nn.Softmax(dim=1)(assignments)

        mixture_weights = []
        means = []
        log_sigmas = []
        for k in range(num_of_modes):
            y_ik = assignments_adjusted[:, :, k]  # bs, 20
            w_k = torch.mean(y_ik, dim=1)  # bs

            mu_k_x = self.average_weighted_norm(samples[:, :, :, 0,], y_ik)  # (bs, 12)
            mu_k_y = self.average_weighted_norm(samples[:, :, :, 1], y_ik)
            mu_k = torch.stack([mu_k_x, mu_k_y], dim=2)  # (bs, 12, 2)

            mu_k_repeated = torch.unsqueeze(mu_k, dim=1).repeat(1, 20, 1, 1)  # (bs, 20, 12, 2)
            diff = (samples - mu_k_repeated) ** 2
            var_k_x = self.average_weighted_norm(diff[:, :, :, 0], y_ik)
            var_k_y = self.average_weighted_norm(diff[:, :, :, 1], y_ik)
            var_k = torch.stack([var_k_x, var_k_y], dim=2)  # (bs, 12, 2)
            sigma_k = torch.sqrt(var_k)
            log_sigma_k = torch.log(sigma_k)

            mixture_weights.append(w_k)
            means.append(mu_k)
            log_sigmas.append(log_sigma_k)

        return means, log_sigmas, mixture_weights

    def adjusted_sigmoid(self, x, min=-6, max=6):
        range = max - min
        x_scaled = x * (4.0 / range)
        sig = torch.sigmoid(x_scaled)
        sig_scaled = sig * range
        if min != 0:
            sig_scaled_shifted = sig_scaled + min
        else:
            sig_scaled_shifted = sig_scaled

        return sig_scaled_shifted

    def decoder_fitting(self, y):
        y_flatten = torch.reshape(y, (y.shape[0], -1)) # bs, 20*12*2
        intermediate_features = self.node_modules[self.node_type + '/fitting_0'](y_flatten)   # bs, 512
        soft_assignments = self.node_modules[self.node_type + '/fitting_1'](intermediate_features)  # bs, 20*4
        soft_assignments = soft_assignments.view(soft_assignments.shape[0], 20, 4)
        # len of the lists is 4 (num of modes), means, log_sigmas elements are (bs, 12, 2), mixture_weights are (bs)
        means, log_sigmas, mixture_weights = self.get_gaussian_mixture_model_from_samples(y, soft_assignments)
        bounded_log_sigmas = [self.adjusted_sigmoid(log_sigmas[i]) for i in range(len(log_sigmas))]
        return means, bounded_log_sigmas, mixture_weights

    def gmm_loss(self, means, log_sigmas, mixture_weights, labels):
        # labels has shape (bs, 12, 2)
        # shape of means list of (bs, 12, 2)
        # shape of log_sigmas list of (bs, 12, 2) for x and y
        # shape of mixture_weights list of (bs)
        num_modes = len(means)
        loss_modes = []
        eps = 1e-5 / 2.0
        for i in range(num_modes):
            diff = (labels - means[i]) ** 2
            sigma = torch.exp(log_sigmas[i])
            sigma_sq_inv = 1.0 / (sigma ** 2 + eps)
            c = torch.sum(diff * sigma_sq_inv, dim=2)  # (bs, 12)
            c_exp = torch.exp(-1 * c)
            sxsy = sigma[:, :, 0] * sigma[:, :, 1]
            final = c_exp / (sxsy + eps)
            final_weighted = final * torch.unsqueeze(mixture_weights[i], dim=1).repeat(1, 12)
            loss_modes.append(final_weighted)
        sum_loss = torch.stack(loss_modes, dim=2)  # bs, 12, 4
        log_nll = -1 * torch.log(torch.sum(sum_loss, dim=(1, 2)) + eps)
        return torch.mean(log_nll)


    def ewta_loss(self, y, labels, mode='epe-all', top_n=1, class_weight=None):
        # y has shape (bs, 20, 12 ,2)
        # labels has shape (bs, 12, 2)
        gts = torch.stack([labels for i in range(20)], dim=1)  # (bs, 20, 12, 2)
        diff = (y - gts) ** 2
        channels_sum = torch.sum(diff, dim=3)  # (bs, 20, 12)
        spatial_epes = torch.sqrt(channels_sum + 1e-20)  # (bs, 20, 12)

        sum_spatial_epe = torch.zeros(spatial_epes.shape[0])
        if mode == 'epe':
            spatial_epe, _ = torch.min(spatial_epes, dim=1)  # (bs, 12)
            sum_spatial_epe = torch.sum(spatial_epe, dim=1)
        elif mode == 'epe-top-n' and top_n > 1:
            spatial_epes_min, _ = torch.topk(-1 * spatial_epes, top_n, dim=1)
            spatial_epes_min = -1 * spatial_epes_min  # (bs, top_n, 12)
            sum_spatial_epe = torch.sum(spatial_epes_min, dim=(1, 2))
        elif mode == 'epe-all':
            sum_spatial_epe = torch.sum(spatial_epes, dim=(1, 2))

        if class_weight is not None:
            sum_spatial_epe = sum_spatial_epe * class_weight

        return torch.mean(sum_spatial_epe)

    def get_mode_top_n(self, loss_type):
        if 'top' in loss_type:
            return 'epe-top-n', int(loss_type.replace('epe-top-',''))
        else:
            return loss_type, 1

    def train_fitting_loss(self,
                           inputs,
                           inputs_st,
                           first_history_indices,
                           labels,
                           labels_st,
                           neighbors,
                           neighbors_edge_value,
                           robot,
                           map,
                           prediction_horizon,
                           ):
        mode = ModeKeys.TRAIN
        with torch.no_grad():
            x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                    inputs=inputs,
                                                    inputs_st=inputs_st,
                                                    labels=labels,
                                                    labels_st=labels_st,
                                                    first_history_indices=first_history_indices,
                                                    neighbors=neighbors,
                                                    neighbors_edge_value=neighbors_edge_value,
                                                    robot=robot,
                                                    map=map)
            y, features = self.decoder(x, n_s_t0, prediction_horizon)
        means, bounded_log_sigmas, mixture_weights = self.decoder_fitting(y)
        loss = self.gmm_loss(means, bounded_log_sigmas, mixture_weights, labels)
        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                   loss,
                                   self.curr_iter)

        return loss

    def train_loss(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon,
                   loss_type,
                   score,
                   class_weight,
                   contrastive=False,
                   factor_con=100,
                   push_away=False,
                   temp=0.1,
                   class_reweight=False) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=labels,
                                                labels_st=labels_st,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        # x has shape (bs, 64) for both history and edge encoding
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        #features = F.normalize(features, dim=1)
        # y has shape (bs, 12*20*2)
        mode, top_n = self.get_mode_top_n(loss_type)
        if class_reweight:
            loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n, class_weight=class_weight)
        else:
            loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)
        if contrastive:
            if push_away:
                con_loss, positive, negative = self.contrastive_three_modes_push_hard_loss(features, score)
            else:
                con_loss, positive, negative = self.contrastive_three_modes_loss(features, score, temp=temp)
            #con_loss, positive, negative = self.sup_con_loss(features, score)
            #con_loss, positive, negative = self.con_future_motion_pattern_loss(features, inputs, labels)
            # factor_dict = {'epe-all': 1 * factor_con,
            #                'epe-top-10': 0.5 * factor_con,
            #                'epe-top-5': 0.25 * factor_con,
            #                'epe-top-2': 0.1 * factor_con,
            #                'epe': 0.05 * factor_con}
            # factor_con = factor_dict[loss_type]
            final_loss = loss + factor_con * con_loss
            #final_loss = con_loss
            if self.log_writer is not None:
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'contrastive_loss'),
                                           con_loss,
                                           self.curr_iter)
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'positives'),
                                           positive,
                                           self.curr_iter)
                self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'negatives'),
                                           negative,
                                           self.curr_iter)
        else:
            final_loss = loss

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                   loss,
                                   self.curr_iter)

        return final_loss

    def train_loss_SimClr(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   neighbors,
                   neighbors_edge_value,
                   robot,
                   map,
                   prediction_horizon,
                   loss_type,
                   score,
                   lambda_kalman=1.0,
                   lambda_sim=1.0) -> torch.Tensor:

        mode = ModeKeys.TRAIN
        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=labels,
                                                labels_st=labels_st,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)
        # x has shape (bs, 64) for both history and edge encoding
        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features_all = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        f1, f2 = torch.split(features_all, [256, 256], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)  # (256, 2, 232)
        # y has shape (bs, 12*20*2)
        mode, top_n = self.get_mode_top_n(loss_type)
        ewta_loss = self.ewta_loss(y, labels, mode=mode, top_n=top_n)
        sim_loss = self.simclr_loss(features, temp=0.5)
        kalman_contrastive_loss, positive, negative = self.contrastive_three_modes_loss(features_all, score, temp=0.5)
        final_loss = ewta_loss + lambda_kalman * kalman_contrastive_loss + lambda_sim * sim_loss
        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'sim_loss'),
                                       sim_loss,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'ewta_loss'),
                                       ewta_loss,
                                       self.curr_iter)

            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'kalman_contrastive_loss'),
                                       kalman_contrastive_loss,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'positives'),
                                       positive,
                                       self.curr_iter)
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'negatives'),
                                       negative,
                                       self.curr_iter)
        return final_loss

    def simclr_loss(self, features, temp=0.5, base_temperature=0.07):
        # features has shape (bs, 2, 232)
        # scores has shape (bs*2)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # (bs*2, 232)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), temp)  # (bs*2, bs*2)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)  # (bs*2, bs*2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # (bs*2, 1)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # (bs*2)
        mean_log_prob_pos = mean_log_prob_pos

        # loss
        loss = - (temp / base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss


    def con_rel_loss(self, features, inputs, labels, temp=0.1, base_temperature=0.07):
        # features has shape (bs,64)
        # inputs has shape (bs, 8, 6)
        # labels has shape (bs, 12, 2)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        last_labels = labels[:,-1,:]  # (bs, 2)
        last_observation = inputs[:, -1, 0:2]  # (bs, 2)
        first_observation = inputs[:, 0, 0:2]  # (bs, 2)
        history_vec = last_observation - first_observation
        future_vec = last_labels - last_observation  # (bs, 2)
        history_vec = F.normalize(history_vec, dim=1)
        future_vec = F.normalize(future_vec, dim=1)
        dot_vel = torch.matmul(history_vec, future_vec.T)  # (bs, bs)
        scores = torch.diagonal(dot_vel).view(-1, 1)  # bs
        mask_positives = (torch.abs(scores.sub(scores.T)) < 0.1).float().to(device)  # (bs,bs)
        mask_negatives = (torch.abs(scores.sub(scores.T)) > 0.5).float().to(device)  # (bs,bs)
        mask_neutral = mask_positives + mask_negatives  # the zeros elements represent the neutral samples

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
        loss = - (temp / base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()

    def con_future_motion_pattern_loss(self, features, inputs, labels, temp=0.1, base_temperature=0.07):
        # features has shape (bs,64)
        # inputs has shape (bs, 8, 6)
        # labels has shape (bs, 12, 2)
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        last_labels = labels[:,-1,:]  # (bs, 2)
        last_observation = inputs[:, -1, 0:2]  # (bs, 2)
        future_vec = last_labels - last_observation  # (bs, 2)
        x = future_vec
        x_norm = (x ** 2).sum(1).view(-1, 1)  # (bs, 1)
        y_t = torch.transpose(x, 0, 1)  # (2, bs)
        y_norm = x_norm.view(1, -1)  # (1, bs)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        dist[dist != dist] = 0  # (bs, bs)

        mask_positives = (dist < 0.05).float().to(device)  # (bs,bs)
        mask_negatives = (dist > 0.5).float().to(device)  # (bs,bs)
        mask_neutral = mask_positives + mask_negatives  # the zeros elements represent the neutral samples

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
        loss = - (temp / base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()

    def contrastive_three_modes_loss(self, features, scores, temp=0.1, base_temperature=0.07):
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
        loss = - (temp / base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()

    def contrastive_three_modes_push_hard_loss(self, features, scores, temp=0.1, base_temperature=0.07):
        # features has shape (bs,64)
        # scores has shape (bs)
        # univ (14_20) 0.1, 0.5 >>> All (0.16,0.33) Challenging (0.32,0.69)
        # univ (15_07) 0.1, 0.7 >>> All (0.16,0.33) Challenging (0.32,0.68)
        # univ (15_44) 0.08,0.7 >>> All (0.16,0.33) Challenging (0.32,0.69)
        # univ (16_26) 0.1, 0.5 (200) >>> All (0.16,0.32) Challenging (0.32,0.69)
        # univ (17_39) 0.1, 0.7 (start: epe-10) >>> All (0.16,0.33) Challenging (0.32,0.70)
        # univ (18_02) 0.1, 0.7 (start: epe-5) >>> All (0.16,0.33) Challenging (0.33,0.72)
        # univ (18_24) 0.1, 0.7 (start: epe-2) >>> All (0.16,0.33) Challenging (0.32,0.70)
        # univ (19_) 0.1, 0.7 (no head) >>> All (0.16,0.32) Challenging (0.32,0.69)  THIS
        # eth base >>> All (0.34, 0.65) Challenging (0.63, 1.31)
        # eth (13_29) 0.1, 0.7 >>> All (0.37, 0.74) Challenging (0.75, 1.78)
        # eth (15_34) 0.2, 0.8 >>> All (0.35, 0.65) Challenging (0.64, 1.32)
        # eth (16_58) 0.1, 1.0 >>> All (0.35, 0.62) Challenging (0.64, 1.33)  THIS
        # eth (17_19) (sup_con_loss 0.5) >>> All (0.35, 0.66) Challenging (0.69, 1.60)
        # eth (17_48) (sup_con_loss with scores (deter vs multi) 0.5) >>> All (0.34, 0.65) Challenging (0.67, 1.45)
        # eth (19_38) push away the hard >>> All (0.35, 0.65) Challenging (0.66, 1.45)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        scores = scores.contiguous().view(-1, 1)  # (bs,1)
        mask_positives = (torch.abs(scores.sub(scores.T)) < 0.1).float().to(device)  # (bs,bs)  # 0.1, 0.08
        mask_negatives = (torch.abs(scores.sub(scores.T)) > 1.0).float().to(device)  # (bs,bs)  # 0.5, 0.7
        mask_neutral = mask_positives + mask_negatives  # the zeros elements represent the neutral samples
        mask_batch = (scores[:, 0] < 2.0).float().to(device)  # bs

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
        mean_log_prob_pos = mean_log_prob_pos * mask_batch

        # loss
        loss = - (temp / base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss, mask_positives.sum(1).mean(), mask_negatives.sum(1).mean()

    def sup_con_loss(self, features, scores, temp=0.1, base_temperature=0.07):
        # features has shape (bs,64)
        # scores has shape (bs)

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0]
        labels = (scores / 0.5).int()  # quantify the scores to get int labels
        #mask_batch = (labels > 10).float().to(device)  # bs
        labels = labels.contiguous().view(-1, 1)  # (bs,1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temp)  # (n,n)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (n,1)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-20)
        # mean_log_prob_pos = mean_log_prob_pos * mask_batch

        # loss
        loss = - (temp / base_temperature) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss, mask.sum(1).mean(), batch_size - mask.sum(1).mean()

    def contrastive_threshold_loss(self, features, scores):
        # features has shape (bs,64)
        # scores has shape (bs)

        batch_size = features.shape[0]
        scores = scores.contiguous().view(-1, 1)  # (bs,1)
        mask = (torch.abs(scores.sub(scores.T)) < 0.3).float().to(self.device)  # (bs,bs)

        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(features, features.T), 0.5)  # (n,n)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  # (n,1)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-20)

        # loss
        loss = - (0.5 / 0.07) * mean_log_prob_pos
        loss = loss.view(1, batch_size).mean()

        return loss, mask.sum(1).mean(), mask.sum(1).mean()

    def eval_loss(self,
                  inputs,
                  inputs_st,
                  first_history_indices,
                  labels,
                  labels_st,
                  neighbors,
                  neighbors_edge_value,
                  robot,
                  map,
                  prediction_horizon) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL

        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=labels,
                                                labels_st=labels_st,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)

        ### Importance sampled NLL estimate
        y, _ = self.decoder(x, n_s_t0, prediction_horizon)
        loss = self.ewta_loss(y, labels)

        return loss

    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon):
        mode = ModeKeys.PREDICT

        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)

        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        features = F.normalize(features, dim=1)
        #features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        return y, features

    def predict_fitting(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon):
        mode = ModeKeys.PREDICT

        x, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                inputs=inputs,
                                                inputs_st=inputs_st,
                                                labels=None,
                                                labels_st=None,
                                                first_history_indices=first_history_indices,
                                                neighbors=neighbors,
                                                neighbors_edge_value=neighbors_edge_value,
                                                robot=robot,
                                                map=map)

        y, features = self.decoder(x, n_s_t0, prediction_horizon)
        means, bounded_log_sigmas, mixture_weights = self.decoder_fitting(y)
        features = F.normalize(features, dim=1)
        #features = F.normalize(self.node_modules[self.node_type + '/con_head'](features), dim=1)
        return y, features, means, bounded_log_sigmas, mixture_weights
