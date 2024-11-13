import torch
import torch.nn as nn
import numpy as np
from prism.agents.models import FFNNModel
# from prism.agents.models.iqn_model import reinit_layer


class QEnsemble(nn.Module):
    def __init__(self, n_input_features, n_actions, n_heads=10, use_layer_norm=True,
                 n_model_layers=0, model_layer_size=0, model_activation=nn.ReLU,
                 q_loss_function=None, ensemble_variation_coef=0, q_loss_weight=1,
                 squish_function=None, unsquish_function=None,
                 use_double_q_learning=True,
                 device="cpu"):

        super().__init__()
        self.device = device
        self.q_loss_weight = q_loss_weight
        self.squish_function = squish_function
        self.unsquish_function = unsquish_function
        self.loss = q_loss_function
        self.use_double_q_learning = use_double_q_learning
        self.ensemble_variation_coef = ensemble_variation_coef
        self.theil = 0

        q_heads = []
        for i in range(n_heads):
            if n_model_layers > 0:

                branch = FFNNModel(n_input_features=n_input_features, n_output_features=n_actions,
                                   n_layers=n_model_layers, layer_width=model_layer_size, use_layer_norm=use_layer_norm,
                                   output_act_fn=None, act_fn=model_activation, device=device)
                q_heads.append(branch)
            else:
                if use_layer_norm:
                    q_heads.append(nn.Sequential(nn.LayerNorm(n_input_features), nn.Linear(n_input_features, n_actions)).to(device))
                else:
                    q_heads.append(nn.Linear(n_input_features, n_actions).to(device))
            # reinit_layer(q_heads[-1].model[-1], self.device, noise_std=0.1)

        self.q_heads = nn.ModuleList(q_heads).to(device)

    def forward(self, x):
        if type(x) is not torch.Tensor:
            if type(x) is not np.array:
                x = np.asarray(x, dtype=np.float32)
            x = torch.from_numpy(x).to(self.device)

        q_outputs = []
        for i in range(len(self.q_heads)):
            q_outputs.append(self.q_heads[i](x))
        return q_outputs

    def get_loss(self, embedded_obs, embedded_next_obs, batch_acts, batch_returns, dones_and_gamma, target_model=None):
        if target_model is None:
            target_model = self

        batch_size = batch_acts.shape[0]
        batch_acts = batch_acts.view(-1)
        online_current_q_estimates = self.forward(embedded_obs)

        with torch.no_grad():
            # No target model, bootstrap entirely from self.
            if target_model is self:
                online_next_q_estimates = self.forward(embedded_next_obs)
                target_next_q_estimates = online_next_q_estimates

            # Double DQN
            elif self.use_double_q_learning:
                online_next_q_estimates = self.forward(embedded_next_obs)
                target_next_q_estimates = target_model.forward(embedded_next_obs)

            # DQN
            else:
                target_next_q_estimates = target_model.forward(embedded_next_obs)
                online_next_q_estimates = target_next_q_estimates

            batch_index_range = torch.arange(start=0, end=batch_size, step=1, device=self.device)
            q_head_targets = []
            for i in range(len(target_next_q_estimates)):
                best_next_action = online_next_q_estimates[i].argmax(dim=-1).view(-1)
                next_action_q_estimate = target_next_q_estimates[i][batch_index_range, best_next_action]

                if self.unsquish_function is not None:
                    next_action_q_estimate = self.unsquish_function(next_action_q_estimate)

                target = batch_returns + next_action_q_estimate * dones_and_gamma
                if self.squish_function is not None:
                    target = self.squish_function(target)
                q_head_targets.append(target)

        q_loss = 0
        n_q_heads = len(self.q_heads)

        if self.ensemble_variation_coef != 0:
            # param_set = []
            l2_set = []
            for i in range(n_q_heads):
                # param_set.append(nn.utils.parameters_to_vector(self.q_heads[i].parameters()))
                l2_set.append(nn.utils.parameters_to_vector(self.q_heads[i].parameters()).norm())
            # params = torch.stack(param_set)
            # mean_params = torch.mean(params, dim=0)
            # vec_dists = torch.norm(params - mean_params, dim=1)
            # theil_index = vec_dists.mean()
            l2_set = torch.stack(l2_set)
            mean_l2 = torch.mean(l2_set)

            # gini_coefficient = 0
            ratio = l2_set / mean_l2
            theil_index = (ratio*torch.log(ratio)).mean()
            # print(theil_index.detach().item(), "|", ratio.detach().tolist())
        else:
            theil_index = 0

        for i in range(n_q_heads):
            q_error = self.loss(online_current_q_estimates[i][batch_index_range, batch_acts], q_head_targets[i])
            q_loss += q_error / n_q_heads

        #     for j in range(i, n_q_heads):
        #         gini_coefficient += torch.abs(l2_set[i] - l2_set[j] / mean_l2)
        # gini_coefficient /= n_q_heads*n_q_heads
        self.theil = theil_index
        return self.q_loss_weight * (q_loss - theil_index*self.ensemble_variation_coef)  # gini_coefficient * 1e-6

    def log(self, logger):
        if type(self.theil) is torch.Tensor:
            logger.log_data(data=self.theil.detach().item(), group_name="Debug/Q Ensemble", var_name="Variation Loss")
