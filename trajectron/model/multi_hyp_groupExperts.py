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
                 train_borders,
                 train_borders_match_class_per_bin, 
                 class_count_dict,
                 group_experts= True,
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
        self.class_count_dict = class_count_dict
        self.train_borders = train_borders
        self.train_borders_match_class_per_bin = train_borders_match_class_per_bin # the classes for each group; 0 is always reserved for others
        self.nb_classes = len(self.class_count_dict[0])
        self.nb_bins = len(self.train_borders.keys()) - 1 # if nb_bins = 4; you have 0,1,2,3,4
        self.group_experts = group_experts
        self.criterion_ldam = LDAMLoss(cls_num_list= [*self.class_count_dict[0].values()])
        self.criterion_con_scores = ScoreBasedConLoss()
        if group_experts:
            nb_observations = sum([*self.class_count_dict[0].values()])
            
            self.criterion_ldam_bins = {}
            self.criterion_ldam_bins_weights = {}
            for curr_bin in range(self.nb_bins+1):
                class_count = [*self.class_count_dict[0].values()]
                curr_class_count = class_count[self.train_borders[curr_bin][0]: self.train_borders[curr_bin][-1] +1]
                
                # we should add number of observation of others; we add it in the beginning because label = 0
                #TODO do it with the proper values
                
                #if curr_bin == 0:
                #    curr_class_count.insert(0, curr_class_count[-1])
                #else:
                #    curr_class_count.insert(0, curr_class_count[0] + 10)
                curr_class_count.insert(0, nb_observations - sum( curr_class_count))

                self.criterion_ldam_bins[curr_bin] = LDAMLoss(cls_num_list= curr_class_count)
                self.criterion_ldam_bins_weights[curr_bin] = curr_class_count

        self.node_modules = dict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
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
        
        
        if self.group_experts:
            self.losses = {}
            for j in range(self.nb_bins + 1):
                self.losses['loss_'+str(j)] = self.train_borders[j]


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

        z_size = self.hyperparams['Enc_FC_dims']
        self.add_submodule(self.node_type + '/EncoderFC',
                           model_if_absent=nn.Linear(x_size, z_size))
        z_size = x_size
        ####################
        #   Decoder LSTM   #
        ####################

        if self.hyperparams['incl_robot_node']:
            decoder_input_dims = self.pred_state_length+ self.robot_state_length + z_size
        else:
            decoder_input_dims = 20 * self.pred_state_length  + z_size

        self.add_submodule(self.node_type + '/decoder/rnn_cell',
                           model_if_absent=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_h',
                           model_if_absent=nn.Linear(z_size, self.hyperparams['dec_rnn_dim']))
        self.add_submodule(self.node_type + '/decoder/initial_mu',
                           model_if_absent=nn.Linear(self.state_length, self.pred_state_length))
        self.add_submodule(self.node_type + '/decoder/kalman_logits',
                           model_if_absent=NormedLinear(self.hyperparams['dec_rnn_dim'] + decoder_input_dims, self.nb_classes))
        self.add_submodule(self.node_type + '/decoder/kalman_logits_groupExperts',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'] + decoder_input_dims, self.nb_classes + self.nb_bins+1))
        
        self.add_submodule(self.node_type + '/decoder/proj_to_mus',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['num_hyp'] * self.pred_state_length))
        self.add_submodule(self.node_type + '/con_head',
                           model_if_absent=nn.Linear(self.hyperparams['dec_rnn_dim'] + decoder_input_dims,
                                                     self.hyperparams['output_con_model']))
        
        self.x_size = x_size
        self.z_size = z_size

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

###########################################################################################
# Annealers
    # def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
    #     value_scheduler = None
    #     rsetattr(self, name + '_scheduler', value_scheduler)
    #     if creation_condition:
    #         annealer_kws['device'] = self.device
    #         value_annealer = annealer(annealer_kws)
    #         rsetattr(self, name + '_annealer', value_annealer)

    #         # This is the value that we'll update on each call of
    #         # step_annealers().
    #         rsetattr(self, name, value_annealer(0).clone().detach())
    #         dummy_optimizer = optim.Optimizer(
    #             [rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
    #         rsetattr(self, name + '_optimizer', dummy_optimizer)

    #         value_scheduler = CustomLR(dummy_optimizer,
    #                                    value_annealer)
    #         rsetattr(self, name + '_scheduler', value_scheduler)

    #     self.schedulers.append(value_scheduler)
    #     self.annealed_vars.append(name)

    # def set_annealing_params(self):
    #     self.schedulers = list()
    #     self.annealed_vars = list()

    #     self.create_new_scheduler(name='kl_weight',
    #                               annealer=sigmoid_anneal,
    #                               annealer_kws={
    #                                   'start': self.hyperparams['kl_weight_start'],
    #                                   'finish': self.hyperparams['kl_weight'],
    #                                   'center_step': self.hyperparams['kl_crossover'],
    #                                   'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
    #                                       'kl_sigmoid_divisor']
    #                               })

    #     self.create_new_scheduler(name='latent.temp',
    #                               annealer=exp_anneal,
    #                               annealer_kws={
    #                                   'start': self.hyperparams['tau_init'],
    #                                   'finish': self.hyperparams['tau_final'],
    #                                   'rate': self.hyperparams['tau_decay_rate']
    #                               })

    #     self.create_new_scheduler(name='latent.z_logit_clip',
    #                               annealer=sigmoid_anneal,
    #                               annealer_kws={
    #                                   'start': self.hyperparams['z_logit_clip_start'],
    #                                   'finish': self.hyperparams['z_logit_clip_final'],
    #                                   'center_step': self.hyperparams['z_logit_clip_crossover'],
    #                                   'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
    #                                       'z_logit_clip_divisor']
    #                               },
    #                               creation_condition=self.hyperparams['use_z_logit_clipping'])

    # def step_annealers(self):
    #     # This should manage all of the step-wise changed
    #     # parameters automatically.
    #     for idx, annealed_var in enumerate(self.annealed_vars):
    #         if rgetattr(self, annealed_var + '_scheduler') is not None:
    #             # First we step the scheduler.
    #             with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
    #                 warnings.simplefilter("ignore")
    #                 rgetattr(self, annealed_var + '_scheduler').step()

    #             # Then we set the annealed vars' value.
    #             rsetattr(self, annealed_var, rgetattr(
    #                 self, annealed_var + '_optimizer').param_groups[0]['lr'])

    #     self.summarize_annealers()

    # def summarize_annealers(self):
    #     if self.log_writer is not None:
    #         for annealed_var in self.annealed_vars:
    #             if rgetattr(self, annealed_var) is not None:
    #                 self.log_writer.add_scalar('%s/%s' % (str(self.node_type), annealed_var.replace('.', '/')),
    #                                            rgetattr(self, annealed_var), self.curr_iter)

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
        #with torch.no_grad():
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
        return x, x_r_t, y_e, y_r, y, n_s_t0

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
        #with torch.no_grad():
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

    def encoder(self, x):
        """
        Encodes: the combined in put into a hidden representation hx
        :param Input / Condition tensor.
        """
        # z = self.node_modules[self.node_type + '/EncoderFC'](x)
        return x

    def decoder(self, z, n_s_t0, x_nr_t, horizon, joint_train, contrastive = False, fix_encoder = False):
        """
        Decoder: produces the hypotheses
        :param h: hidden representation
        """
        """
        For time step in Horizon, Cell inputs:
            - input: concatenate the closes hypothesis to ground truth with z => [mu, z]
            - hidden state: (output of the cell => mus ie. num hypothesis)
            - output is the next hidden state => mus of next hypothesis
        TODO: 
        - Mu must have shape (bs, num_hyp, pred_state) to match the ground truth.
            - we can predict 2*num_hyp.
            - To compare we must use the predicted y after integrator.
        """
        cell = self.node_modules[self.node_type + '/decoder/rnn_cell']
        initial_h_model = self.node_modules[self.node_type + '/decoder/initial_h']
        initial_mu_model = self.node_modules[self.node_type + '/decoder/initial_mu']
        logits_model = self.node_modules[self.node_type + '/decoder/kalman_logits']
        logits_model_groupExperts = self.node_modules[self.node_type + '/decoder/kalman_logits_groupExperts']
        project_to_mus = self.node_modules[self.node_type + '/decoder/proj_to_mus']
        con_model = self.node_modules[self.node_type + '/con_head']

        initial_h = initial_h_model(z)
        initial_mu = initial_mu_model(n_s_t0)  # [bs, num_hyp *2]
        if fix_encoder: 
            initial_h = F.relu(initial_h)
            initial_mu = F.relu(initial_mu)

        if self.hyperparams['incl_robot_node']:
            input_ = torch.cat([z, initial_mu.repeat(1, 20), x_nr_t], dim=1)
        else:
            #input_ = torch.cat([z, initial_mu.repeat(1, 20)], dim=1)
            input_ = torch.cat([z, initial_mu.repeat(1, 20)], dim=1)

        features = torch.cat([input_, initial_h], dim=1)
        

        if self.group_experts:
            logits = logits_model_groupExperts(features)
        else:
            logits = logits_model(features)
        
        if contrastive:
            features = con_model(features)
        
        if joint_train == False:
            # ldam classification
            return logits, F.normalize(features , dim = 1)
        else:
            # joint optimization
            h = initial_h
            mus = []
            #import pdb; pdb.set_trace()
            for t in range(horizon):
                h_state = cell(input_, h)  # [bs, rnn_hidden_shape]
                raw_mus = project_to_mus(h_state)
                mu_x = raw_mus[:, self.hyperparams['num_hyp']:]
                mu_y = raw_mus[:, :self.hyperparams['num_hyp']]
                pred_mu = torch.stack([mu_x, mu_y], axis=2)  # [bs,num_hyp, 2]
                mus.append(pred_mu)
                if self.hyperparams['incl_robot_node']:
                    input_ = torch.cat([z, raw_mus, x_nr_t], dim=1)
                else:
                    input_ = torch.cat([z, raw_mus], dim=1)
                h = h_state
            hypothesis = torch.stack(mus, dim=2)  # [bs, num_hyp, horizon, 2]
            hypothesis = self.dynamic.integrate_samples(hypothesis, None)  # [bs, num_hyp, horizon, 2]
            return hypothesis, F.normalize(features , dim = 1), logits


    def closest_mu(self, mus, gt):
        gt = gt.unsqueeze(1).repeat(1, self.hyperparams['num_hyp'], 1)
        dist = torch.sum((mus - gt)**2, dim=2)
        dist = torch.sqrt(dist)
        _, min_indices = torch.min(dist, dim=1)
        # here min_indices has shape [bs]
        # We need to collect from every row the minimum using indices and produce [bs, 2] tensor.
        # https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4
        min_indices = min_indices.unsqueeze(1).unsqueeze(1).repeat(1, 1, 2)
        return mus.gather(1, min_indices).squeeze()

    def wta(self, gt, hyp):
        """
        Calculates the WTA Loss
        :param gt [bs, horizon, 2]
        :param hyp [bs, n_hyp, horizon, 2]
        """
        eps = 0.001
        # gt -> [bs, num_hyp, horizon, 2]
        gt = gt.unsqueeze(1).repeat(1, self.hyperparams['num_hyp'], 1, 1)
        dist = torch.sum((hyp - gt)**2, dim=3)  # [bs, num_hyp, horizon]
        dist = torch.sqrt(dist + eps)  # [bs, num_hyp, horizon]
        min_loss, _ = torch.min(dist, dim=1)  # [bs, horizon]
        # sum over horizon and mean over batch
        losses = torch.sum(min_loss, dim=1).mean(dim=0)
        return losses

    def ewta(self, gt, hyp, top_n=8):
        """
        Calculates the Evolved WTA Loss
        :param gt [bs, horizon, 2]
        :param hyp [bs, n_hyp, horizon, 2]
        :param top_n initially num hyp (according to the paper)
        """
        sum_losses = 0.0
        eps = 0.001
        # gt -> [bs, num_hyp, horizon, 2]
        gt = gt.unsqueeze(1).repeat(1, self.hyperparams['num_hyp'], 1, 1)
        dist = torch.sum((hyp - gt)**2, dim=3)  # [bs, num_hyp, horizon]
        dist = torch.sqrt(dist + eps)
        # [bs, top_k, horizon]
        min_loss, _ = torch.topk(dist, k=top_n, dim=1, largest=False, sorted=True)
        for i in range(top_n):
            # sum over the horizon of the trajectory of ith hyp
            losses = torch.sum(min_loss[:, i, :], dim=1)
            losses = losses.mean(dim=0)
            sum_losses += losses
        return sum_losses

    def get_accuracy(self, curr_nodes_bin_batch, curr_targets_cl_bin_batch) :
        accuracy_bin = 0
        y_hat = F.log_softmax(curr_nodes_bin_batch, dim = 1).max(1)[1]
        accuracy_bin = torch.sum(y_hat == curr_targets_cl_bin_batch).item() / len(curr_targets_cl_bin_batch)
        return round(accuracy_bin * 100, 2)


    def get_losses_class_group(self, y_hat_cl, targets_cl, weight, target_class_bin, others_factor = 8):
        current_losses = {}
        current_losses_no_reweight = {}
        accuracies_bins = {}
        #self.train_borders 
        #self.train_borders_match_class_per_bin 
        curr_start_nodes = 0
        for curr_bin in range(self.nb_bins + 1):
            curr_indices = torch.where(target_class_bin == curr_bin)[0]
            other_curr_indices = torch.where(target_class_bin != curr_bin)[0]
            if len(curr_indices)== 0:
                current_losses[curr_bin] = None
                current_losses_no_reweight[curr_bin] = None
                accuracies_bins[curr_bin] = None
            else:
                curr_targets_cl = targets_cl[curr_indices]
                curr_targets_cl_bin = torch.zeros_like(curr_targets_cl)
                
                for i in range(len(curr_targets_cl)):
                    curr_targets_cl_bin[i] = self.train_borders_match_class_per_bin[curr_bin][self.train_borders[curr_bin].index(curr_targets_cl[i])]
                #curr_targets_cl_bin are labels of curr_nodes_not_others examples 
                curr_nodes = y_hat_cl[:, curr_start_nodes:curr_start_nodes +self.train_borders_match_class_per_bin[curr_bin][-1]+1]
                curr_nodes_not_others = curr_nodes[curr_indices]
                if curr_bin == 0:
                    curr_nodes_others = curr_nodes[other_curr_indices]
                else:
                    if len(other_curr_indices) >= len(curr_indices) * others_factor:
                        perm = torch.randperm(len(other_curr_indices))
                        idx = perm[:len(curr_indices) * others_factor]
                        other_curr_indices = other_curr_indices[idx]

                    curr_nodes_others = curr_nodes[other_curr_indices]
                #curr_targets_cl_bin_others are labels of curr_nodes_others examples ==> all labels of others are 0 
                curr_targets_cl_bin_others = torch.zeros(len(curr_nodes_others), dtype=torch.int64).to(self.device)
                curr_targets_cl_bin_batch = torch.cat((curr_targets_cl_bin, curr_targets_cl_bin_others), 0)
                curr_nodes_bin_batch = torch.cat((curr_nodes_not_others, curr_nodes_others), 0)
                assert(len(curr_targets_cl_bin_batch) == len(curr_nodes_bin_batch))
                # we should randomize the batch
                permuted_indices = torch.randperm(len(curr_nodes_bin_batch))
                curr_targets_cl_bin_batch = curr_targets_cl_bin_batch[permuted_indices]
                curr_nodes_bin_batch = curr_nodes_bin_batch[permuted_indices]
                # curr_targets_cl_bin_batch qre targets of  curr_nodes_bin_batch
                
                # calculate the loss function of the current bin
                if weight == True:

                    curr_loss = self.criterion_ldam_bins[curr_bin].forward(curr_nodes_bin_batch, curr_targets_cl_bin_batch, weight = self.criterion_ldam_bins_weights[curr_bin])
                else:
                    curr_loss = self.criterion_ldam_bins[curr_bin].forward(curr_nodes_bin_batch, curr_targets_cl_bin_batch, weight=None)
                
                curr_loss_no_reweight = self.criterion_ldam_bins[curr_bin].forward(curr_nodes_bin_batch, curr_targets_cl_bin_batch, weight = None).detach()
                current_losses[curr_bin] = curr_loss.mean()
                current_losses_no_reweight[curr_bin] = curr_loss_no_reweight.mean()

                accuracies_bins[curr_bin] = self.get_accuracy(curr_nodes_bin_batch, curr_targets_cl_bin_batch) 

                curr_start_nodes += self.train_borders_match_class_per_bin[curr_bin][-1]+1
        return current_losses, current_losses_no_reweight, accuracies_bins    



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
                   top_n, 
                   targets_cl,
                   weight,
                   joint_train) -> torch.Tensor:
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
        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)
        z = self.encoder(x)
        if joint_train == False:
            # just classification with LDAM
            y_hat, features = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train)
            loss = self.criterion_ldam.forward(y_hat, targets_cl, weight)
            loss_no_reweighted = self.criterion_ldam.forward(y_hat, targets_cl, weight= None).detach()

            #y_hat_output = self.criterion.get_output(y_hat, targets).detach() # just to calculate the accuracies correctly
            return loss, loss_no_reweighted, F.log_softmax(y_hat, dim = 1).max(1)[1]
        else:
            y_hat_reg, features, y_hat_cl = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train)
            loss_cl = self.criterion_ldam.forward(y_hat_cl, targets_cl, weight)
            loss_cl_no_reweighted = self.criterion_ldam.forward(y_hat_cl, targets_cl, weight= None).detach()
            loss_reg = self.ewta(labels, y_hat_reg, top_n)
            return loss_reg, loss_cl, loss_cl_no_reweighted, F.log_softmax(y_hat_cl, dim = 1).max(1)[1]
    
    def train_lossGroupExperts_con(self,
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
                   top_n, 
                   scores,
                   joint_train) -> torch.Tensor:
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
        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)
        z = self.encoder(x)
        if joint_train == False:
            # just classification with Con scores
            _, features = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train, contrastive = True)
            loss = self.criterion_con_scores.forward(features, scores)

            return loss
        else:
            y_hat_reg, features, _ = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train, contrastive = True)
            loss_cl,_,_ = self.criterion_con_scores.forward(features, scores)
            loss_reg = self.ewta(labels, y_hat_reg, top_n)
            return loss_reg, loss_cl
    

    def train_lossGroupExperts(self,
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
                   top_n, 
                   targets_cl,
                   nb_bins, 
                   target_class_bin,
                   weight,
                   joint_train) -> torch.Tensor:
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
        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=labels,
                                                                     labels_st=labels_st,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)
        z = self.encoder(x)
        if joint_train == False:
            # just classification with LDAM
            y_hat, features = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train)
            loss = self.criterion_ldam.forward(y_hat, targets_cl, weight)
            loss_no_reweighted = self.criterion_ldam.forward(y_hat, targets_cl, weight= None).detach()

            #y_hat_output = self.criterion.get_output(y_hat, targets).detach() # just to calculate the accuracies correctly
            
            return loss, loss_no_reweighted, F.log_softmax(y_hat, dim = 1).max(1)[1]
        
        else:
            y_hat_reg, features, y_hat_cl = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train)
            loss_reg = self.ewta(labels, y_hat_reg, top_n)
            loss_cl_bins ,loss_cl_no_reweighted_bins, accuracies_bins = self.get_losses_class_group(y_hat_cl, targets_cl, weight, target_class_bin)
            loss_cl = 0 
            loss_cl_no_reweighted = 0 
            for curr_bin in range(self.nb_bins + 1):
                if loss_cl_bins [curr_bin] is not None:
                    loss_cl += loss_cl_bins [curr_bin]
                    loss_cl_no_reweighted += loss_cl_no_reweighted_bins[curr_bin]
            #import pdb; pdb.set_trace()
            return loss_reg, loss_cl_bins, loss_cl_no_reweighted_bins, accuracies_bins #F.log_softmax(y_hat_cl, dim = 1).max(1)[1]
    
    
    def eval_lossGroupExperts(self,
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
                   targets_cl,
                   nb_bins, 
                   target_class_bin,
                   weight,
                   joint_train) -> torch.Tensor:
        
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
        with torch.no_grad():
            x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                        inputs=inputs,
                                                                        inputs_st=inputs_st,
                                                                        labels=labels,
                                                                        labels_st=labels_st,
                                                                        first_history_indices=first_history_indices,
                                                                        neighbors=neighbors,
                                                                        neighbors_edge_value=neighbors_edge_value,
                                                                        robot=robot,
                                                                        map=map)
            
            z = self.encoder(x)
            if joint_train == False:
                y_hat, features = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon)
                loss = self.wta(labels, y_hat)
                return loss
            else:
                y_hat_reg, features, y_hat_cl = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train)
                loss_reg = self.wta(labels, y_hat_reg)
                loss_cl_bins ,loss_cl_no_reweighted_bins, accuracies_bins = self.get_losses_class_group(y_hat_cl, targets_cl, weight, target_class_bin)
                loss_cl = 0 
                loss_cl_no_reweighted = 0 
                for curr_bin in range(self.nb_bins + 1):
                    if loss_cl_bins [curr_bin] is not None:
                        loss_cl += loss_cl_bins [curr_bin]
                        loss_cl_no_reweighted += loss_cl_no_reweighted_bins[curr_bin]
                #import pdb; pdb.set_trace()
                return loss_reg, loss_cl_bins, loss_cl_no_reweighted_bins, accuracies_bins #F.log_softmax(y_hat_cl, dim = 1).max(1)[1]
                
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
                  prediction_horizon,
                  targets_cl,
                  weight,
                  joint_train) -> torch.Tensor:
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
        with torch.no_grad():
            x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                        inputs=inputs,
                                                                        inputs_st=inputs_st,
                                                                        labels=labels,
                                                                        labels_st=labels_st,
                                                                        first_history_indices=first_history_indices,
                                                                        neighbors=neighbors,
                                                                        neighbors_edge_value=neighbors_edge_value,
                                                                        robot=robot,
                                                                        map=map)
            
            z = self.encoder(x)
            if joint_train == False:
                y_hat, features = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon)
                loss = self.wta(labels, y_hat)
                return loss
            else:
                y_hat_reg, features, y_hat_cl = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train)
                loss_cl = self.criterion_ldam.forward(y_hat_cl, targets_cl, weight)
                loss_cl_no_reweighted = self.criterion_ldam.forward(y_hat_cl, targets_cl, weight= None).detach()
                loss_reg = self.wta(labels, y_hat_reg)
                return loss_reg, loss_cl, loss_cl_no_reweighted, F.log_softmax(y_hat_cl, dim = 1).max(1)[1]
        


    def predict(self,
                inputs,
                inputs_st,
                first_history_indices,
                neighbors,
                neighbors_edge_value,
                robot,
                map,
                prediction_horizon,
                num_samples,
                joint_train,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False):
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
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT
        x, x_nr_t, y_e, y_r, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                                     inputs=inputs,
                                                                     inputs_st=inputs_st,
                                                                     labels=None,
                                                                     labels_st=None,
                                                                     first_history_indices=first_history_indices,
                                                                     neighbors=neighbors,
                                                                     neighbors_edge_value=neighbors_edge_value,
                                                                     robot=robot,
                                                                     map=map)
        z = self.encoder(x)
        y_pred_reg, features, y_pred_cl = self.decoder(z, n_s_t0, x_nr_t, prediction_horizon, joint_train)
        return y_pred_reg, F.normalize(features , dim = 1), F.log_softmax(y_pred_cl, dim = 1).max(1)[1]


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
