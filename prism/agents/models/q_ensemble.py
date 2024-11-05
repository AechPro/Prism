import torch
import torch.nn as nn
import numpy as np
from prism.agents.models import FFNNModel


class QEnsemble(nn.Module):
    def __init__(self, n_input_features, n_actions, n_heads=10, use_layer_norm=True,
                 n_model_layers=0, model_layer_size=0, model_activation=nn.ReLU,
                 q_loss_function=None, ensemble_variation_coef=0,
                 squish_function=None, unsquish_function=None,
                 use_double_q_learning=True, q_loss_weight=1,
                 device="cpu"):

        super().__init__()
        self.device = device
        self.squish_function = squish_function
        self.unsquish_function = unsquish_function
        self.loss = q_loss_function
        self.use_double_q_learning = use_double_q_learning
        self.ensemble_variation_coef = ensemble_variation_coef
        self.theil = 0
        self.n_heads = n_heads
        self.q_loss_weight = q_loss_weight

        if n_model_layers > 0:
            self.q_heads = nn.ModuleList([
                FFNNModel(n_input_features=n_input_features, n_output_features=n_actions,
                          n_layers=n_model_layers, layer_width=model_layer_size, use_layer_norm=use_layer_norm,
                          output_act_fn=None, act_fn=model_activation, device=device)
                for _ in range(n_heads)
            ])
        else:
            self.q_heads = nn.ModuleList([
                nn.Sequential(nn.LayerNorm(n_input_features), nn.Linear(n_input_features, n_actions))
                if use_layer_norm else nn.Linear(n_input_features, n_actions)
                for _ in range(n_heads)
            ]).to(device)

        self.batch_index_range = None
        self.n_heads_arange = torch.arange(self.n_heads).unsqueeze(1).to(self.device)

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)

        return torch.stack([head(x) for head in self.q_heads])

    def get_loss(self, embedded_obs, embedded_next_obs, batch_acts, batch_returns, dones_and_gamma, target_model=None):
        if target_model is None:
            target_model = self

        batch_size = batch_acts.shape[0]
        batch_acts = batch_acts.view(-1)

        if self.batch_index_range is None or self.batch_index_range.shape[0] != batch_size:
            self.batch_index_range = torch.arange(batch_size, device=self.device)

        online_current_q_estimates = self.forward(embedded_obs)

        with torch.no_grad():
            online_next_q_estimates = self.forward(embedded_next_obs)
            target_next_q_estimates = online_next_q_estimates if target_model is self else target_model.forward(
                embedded_next_obs)

            best_next_actions = online_next_q_estimates.argmax(dim=-1)
            next_action_q_estimates = target_next_q_estimates[self.n_heads_arange, self.batch_index_range, best_next_actions]

            if self.unsquish_function is not None:
                next_action_q_estimates = self.unsquish_function(next_action_q_estimates)

            targets = batch_returns + next_action_q_estimates * dones_and_gamma
            if self.squish_function is not None:
                targets = self.squish_function(targets)

        if self.ensemble_variation_coef != 0:
            l2_set = torch.stack([nn.utils.parameters_to_vector(head.parameters()).norm() for head in self.q_heads])
            mean_l2 = l2_set.mean()
            ratio = l2_set / (mean_l2 + 1e-12)
            theil_index = (ratio * torch.log(ratio)).mean()

            # l2_set = []
            # for i in range(self.n_heads):
            #     l2_set.append(nn.utils.parameters_to_vector(self.q_heads[i].parameters()).norm())
            # l2_set = torch.stack(l2_set)
            # mean_l2 = torch.mean(l2_set)
            #
            # ratio = l2_set / mean_l2
            # theil_index = (ratio * torch.log(ratio)).mean()

        else:
            theil_index = 0

        q_loss = self.loss(online_current_q_estimates[:, self.batch_index_range, batch_acts], targets).mean()

        self.theil = theil_index
        loss = q_loss - theil_index * self.ensemble_variation_coef

        return loss * self.q_loss_weight

    def log(self, logger):
        if isinstance(self.theil, torch.Tensor):
            logger.log_data(data=self.theil.detach().item(), group_name="Debug/Q Ensemble", var_name="Variation Loss")