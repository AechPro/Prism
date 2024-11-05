import torch.nn as nn
import torch
import numpy as np


class IQNModel(nn.Module):
    def __init__(self, n_input_features, n_actions, n_basis_elements, use_layer_norm,
                 n_model_layers=0, model_layer_size=0, model_activation=nn.ReLU,
                 squish_function=None, unsquish_function=None, huber_k=1.0,
                 n_current_quantile_samples=8, n_next_quantile_samples=8,
                 n_quantile_samples_per_action=200, distributional_loss_weight=1,
                 use_double_q_learning=True, propagate_grad=True, risk_policy=None, device="cpu"):

        super().__init__()
        self.device = device
        self.n_actions = n_actions
        self.propagate_grad = propagate_grad
        self.n_current_quantile_samples = n_current_quantile_samples
        self.n_next_quantile_samples = n_next_quantile_samples
        self.n_quantile_samples_per_action = n_quantile_samples_per_action
        self.squish_function = squish_function
        self.unsquish_function = unsquish_function
        self.huber_k = huber_k
        self.use_double_q_learning = use_double_q_learning
        self.distributional_loss_weight = distributional_loss_weight

        self.cos_basis_range = torch.arange(1, n_basis_elements + 1, device=device).unsqueeze(0) * np.pi
        self.risk_policy = risk_policy
        self.phi = nn.Sequential(nn.Linear(n_basis_elements, n_input_features), nn.ReLU()).to(device)

        self.model = None
        if n_model_layers > 0:
            from prism.agents.models import FFNNModel
            self.model = FFNNModel(n_input_features=n_input_features, n_output_features=model_layer_size,
                                   n_layers=n_model_layers, layer_width=model_layer_size, use_layer_norm=use_layer_norm,
                                   output_act_fn=model_activation, act_fn=model_activation, device=device)
            n_input_features = model_layer_size

        if use_layer_norm:
            self.embedding_to_quantile_layer = nn.Sequential(nn.LayerNorm(n_input_features),
                                                             nn.Linear(n_input_features, n_actions)).to(device)
        else:
            self.embedding_to_quantile_layer = nn.Linear(n_input_features, n_actions, device=device)

    def forward(self, x, n_quantile_samples=None, for_action=False, static_quantiles=None):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if not self.propagate_grad:
            x = x.detach()

        x = x.view(x.shape[0], -1)

        n_quantile_samples = self.n_quantile_samples_per_action if for_action else n_quantile_samples

        if static_quantiles is None:
            quantiles = torch.rand(n_quantile_samples * x.shape[0], 1, device=self.device)
        else:
            quantiles = static_quantiles

        embedded_state_tiled = x.repeat_interleave(n_quantile_samples, dim=0)
        embedded_quantiles_tiled = self._embed_quantiles(quantiles)

        if self.risk_policy is not None:
            quantiles = self.risk_policy(quantiles)
            embedded_quantiles_tiled = self._embed_quantiles(quantiles)

        input_to_q_layer = embedded_quantiles_tiled * embedded_state_tiled

        if self.model is not None:
            input_to_q_layer = self.model(input_to_q_layer)
        action_values_at_quantiles = self.embedding_to_quantile_layer(input_to_q_layer)

        if for_action:
            return action_values_at_quantiles.view(n_quantile_samples, -1, self.n_actions)

        return action_values_at_quantiles, quantiles

    def _embed_quantiles(self, quantiles):
        cos_quantiles = torch.cos(quantiles * self.cos_basis_range)
        return self.phi(cos_quantiles)

    def get_loss(self, embedded_obs, embedded_next_obs, batch_acts, batch_returns, dones_and_gamma, target_model=None):
        if not self.propagate_grad:
            embedded_obs = embedded_obs.detach()
            embedded_next_obs = embedded_next_obs.detach()

        target_model = self if target_model is None else target_model
        batch_size = batch_acts.shape[0]

        online_current_quantile_estimates, tau = self.forward(embedded_obs,
                                                              n_quantile_samples=self.n_current_quantile_samples)

        tiled_batch_returns = batch_returns.repeat_interleave(self.n_next_quantile_samples).unsqueeze(1)
        tiled_dones_and_gamma = dones_and_gamma.repeat_interleave(self.n_next_quantile_samples).unsqueeze(1)

        with torch.no_grad():
            online_next_quantile_estimates, _ = self.forward(embedded_next_obs,
                                                             n_quantile_samples=self.n_next_quantile_samples)

            if self.use_double_q_learning and target_model is not self:
                target_next_quantile_estimates, _ = target_model.forward(embedded_next_obs,
                                                                         n_quantile_samples=self.n_next_quantile_samples)
            else:
                target_next_quantile_estimates = online_next_quantile_estimates

            mean_next_online_values = online_next_quantile_estimates.view(self.n_next_quantile_samples, batch_size,
                                                                          -1).mean(dim=0)
            best_next_actions = mean_next_online_values.argmax(dim=-1)

            batch_next_action_quantile_estimates = target_next_quantile_estimates.gather(1,
                                                                                         best_next_actions.repeat_interleave(
                                                                                             self.n_next_quantile_samples).unsqueeze(
                                                                                             1))

            if self.unsquish_function is not None:
                batch_next_action_quantile_estimates = self.unsquish_function(batch_next_action_quantile_estimates)

            target_q = tiled_batch_returns + batch_next_action_quantile_estimates * tiled_dones_and_gamma

            if self.squish_function is not None:
                target_q = self.squish_function(target_q)

            target_q = target_q.view(batch_size, self.n_next_quantile_samples, 1)

        batch_action_quantile_estimates = online_current_quantile_estimates.gather(1, batch_acts.repeat_interleave(
            self.n_current_quantile_samples).unsqueeze(1))
        pred_q = batch_action_quantile_estimates.view(batch_size, self.n_current_quantile_samples, 1)

        # Correct the dimensions here
        deltas = target_q.unsqueeze(2) - pred_q.unsqueeze(
            1)  # Shape: [batch_size, n_next_quantile_samples, n_current_quantile_samples, 1]
        huber_loss = torch.where(deltas.abs() <= self.huber_k, 0.5 * deltas.square(),
                                 self.huber_k * (deltas.abs() - 0.5 * self.huber_k))

        # Correct the dimensions of replay_quantiles
        replay_quantiles = tau.view(batch_size, self.n_current_quantile_samples, 1).repeat(1, 1,
                                                                                           self.n_next_quantile_samples).permute(
            0, 2, 1)
        replay_quantiles = replay_quantiles.unsqueeze(
            -1)  # Shape: [batch_size, n_next_quantile_samples, n_current_quantile_samples, 1]

        quantile_huber_loss = (torch.abs(replay_quantiles - (deltas < 0).float()) * huber_loss) / self.huber_k

        quantile_loss = quantile_huber_loss.sum(dim=(1, 2)).mean()

        return quantile_loss * self.distributional_loss_weight

    def log(self, logger):
        pass
