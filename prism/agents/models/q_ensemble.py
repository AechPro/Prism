import torch
import torch.nn as nn
from prism.agents.models import FFNNModel


class QEnsemble(nn.Module):
    def __init__(self, n_input_features, n_actions, n_heads=10, use_layer_norm=True,
                 n_model_layers=0, model_layer_size=0, model_activation=nn.ReLU,
                 q_loss_function=None, ensemble_variation_coef=0, q_loss_weight=1,
                 squish_function=None, unsquish_function=None,
                 use_double_q_learning=True, sparse_init_p=0.0,
                 device="cpu"):

        super().__init__()
        self.device = device
        self.q_loss_weight = q_loss_weight
        self.squish_function = squish_function
        self.unsquish_function = unsquish_function
        self.loss = q_loss_function
        self.use_double_q_learning = use_double_q_learning
        self.ensemble_variation_coef = ensemble_variation_coef
        self.theil = torch.tensor(0.0, device=device)
        self.n_heads = n_heads

        q_heads = []
        for _ in range(n_heads):
            if n_model_layers > 0:
                branch = FFNNModel(n_input_features=n_input_features, n_output_features=n_actions,
                                   n_layers=n_model_layers, layer_width=model_layer_size, use_layer_norm=use_layer_norm,
                                   sparse_init_p=sparse_init_p, output_act_fn=None, act_fn=model_activation, device=device)

                q_heads.append(branch)
            else:
                if use_layer_norm:
                    q_heads.append(
                        nn.Sequential(nn.LayerNorm(n_input_features), nn.Linear(n_input_features, n_actions)))
                else:
                    q_heads.append(nn.Linear(n_input_features, n_actions))

        self.q_heads = nn.ModuleList(q_heads).to(device)
        self.batch_index_range = None
        self.q_head_index_range = torch.arange(self.n_heads, device=self.device)[None, :]

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        return torch.stack([head(x) for head in self.q_heads], dim=-1)

    def get_loss(self, embedded_obs, embedded_next_obs, batch_acts, batch_returns, dones_and_gamma, target_model=None):
        if target_model is None:
            target_model = self

        batch_size = batch_acts.shape[0]
        if self.batch_index_range is None or self.batch_index_range.shape[0] != batch_size:
            self.batch_index_range = torch.arange(batch_size, device=self.device)

        batch_acts = batch_acts.view(-1)
        online_current_q_estimates = self.forward(embedded_obs)

        with torch.no_grad():
            if target_model is self:
                online_next_q_estimates = target_next_q_estimates = self.forward(embedded_next_obs)
            elif self.use_double_q_learning:
                online_next_q_estimates = self.forward(embedded_next_obs)
                target_next_q_estimates = target_model.forward(embedded_next_obs)
            else:
                target_next_q_estimates = online_next_q_estimates = target_model.forward(embedded_next_obs)

            best_next_action = online_next_q_estimates.argmax(dim=-2)
            next_action_q_estimate = target_next_q_estimates[
                self.batch_index_range[:, None],  # Shape: (B, 1)
                best_next_action,  # Shape: (B, N)
                self.q_head_index_range  # Shape: (1, N)
            ]

            if self.unsquish_function is not None:
                next_action_q_estimate = self.unsquish_function(next_action_q_estimate)

            target = batch_returns.view(-1, 1) + next_action_q_estimate * dones_and_gamma.view(-1, 1)
            if self.squish_function is not None:
                target = self.squish_function(target)

        q_loss = self.loss(online_current_q_estimates[self.batch_index_range, batch_acts, :], target).mean(dim=-1)

        if self.ensemble_variation_coef != 0:
            l2_set = torch.stack([nn.utils.parameters_to_vector(head.parameters()).norm() for head in self.q_heads])
            mean_l2 = l2_set.mean()
            ratio = l2_set / mean_l2
            self.theil = (ratio * torch.log(ratio)).mean()

        return self.q_loss_weight * (q_loss - self.theil * self.ensemble_variation_coef)

    def log(self, logger):
        logger.log_data(data=self.theil.item(), group_name="Debug/Q Ensemble", var_name="Variation Loss")
