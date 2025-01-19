import torch.nn as nn
import torch
import numpy as np


class IQNModel(nn.Module):
    def __init__(self, n_input_features, n_actions, n_basis_elements, use_layer_norm,
                 n_model_layers=0, model_layer_size=0, model_activation=nn.ReLU,
                 squish_function=None, unsquish_function=None, huber_k=1.0,
                 n_current_quantile_samples=8, n_next_quantile_samples=8,
                 n_quantile_samples_per_action=200, use_double_q_learning=True,
                 distributional_loss_weight=1, sparse_init_p=0.0,
                 propagate_grad=True, risk_policy=None, device="cpu"):

        super().__init__()
        self.device = device
        self.n_actions = n_actions
        self.distributional_loss_weight = distributional_loss_weight
        self.propagate_grad = propagate_grad
        self.n_current_quantile_samples = n_current_quantile_samples
        self.n_next_quantile_samples = n_next_quantile_samples
        self.n_quantile_samples_per_action = n_quantile_samples_per_action
        self.squish_function = squish_function
        self.unsquish_function = unsquish_function
        self.huber_k = huber_k
        self.use_double_q_learning = use_double_q_learning

        self.cos_basis_range = torch.arange(1, n_basis_elements + 1, 1, device=device, requires_grad=False)
        self.risk_policy = risk_policy
        self.phi = nn.Sequential(nn.Linear(n_basis_elements, n_input_features), nn.ReLU()).to(device)

        self.model = None
        if n_model_layers > 0:
            from prism.agents.models import FFNNModel
            self.model = FFNNModel(n_input_features=n_input_features, n_output_features=model_layer_size,
                                   n_layers=n_model_layers, layer_width=model_layer_size, use_layer_norm=use_layer_norm,
                                   output_act_fn=model_activation, act_fn=model_activation, device=device,
                                   sparse_init_p=sparse_init_p, use_p_norm=True)

            n_input_features = model_layer_size

        if use_layer_norm:
            self.embedding_to_quantile_layer = nn.Sequential(nn.LayerNorm(n_input_features),
                                                             nn.Linear(n_input_features, n_actions)).to(device)
        else:
            self.embedding_to_quantile_layer = nn.Linear(n_input_features, n_actions, device=device)

    def forward(self, x, n_quantile_samples=None, for_action=False, static_quantiles=None):
        if type(x) is not torch.Tensor:
            if type(x) is not np.array:
                x = np.asarray(x, dtype=np.float32)
            x = torch.from_numpy(x).to(self.device).float()
        else:
            x = x.float().to(self.device)

        if not self.propagate_grad:
            x = x.detach()

        x = x.view(x.shape[0], -1)

        if for_action:
            n_quantile_samples = self.n_quantile_samples_per_action

        if static_quantiles is None:
            quantiles_shape = [n_quantile_samples * x.shape[0], 1]
            quantiles = torch.rand(quantiles_shape, device=self.device).float()
        else:
            quantiles = static_quantiles

        embedded_state_tiled = torch.tile(x, [n_quantile_samples, 1])
        embedded_quantiles_tiled = self._embed_quantiles(quantiles)

        if self.risk_policy is None:
            input_to_q_layer = embedded_quantiles_tiled * embedded_state_tiled
        else:
            distorted_quantiles = self.risk_policy(quantiles)
            embedded_risk_quantiles_tiled = self.embed_quantiles(distorted_quantiles)
            input_to_q_layer = embedded_risk_quantiles_tiled * embedded_state_tiled

        if self.model is not None:
            input_to_q_layer = self.model(input_to_q_layer)
        action_values_at_quantiles = self.embedding_to_quantile_layer(input_to_q_layer)

        if for_action:
            return action_values_at_quantiles.view(n_quantile_samples, -1, self.n_actions)

        return action_values_at_quantiles, quantiles

    def _embed_quantiles(self, quantiles):
        original_tiled_quantiles = torch.tile(quantiles, [1, len(self.cos_basis_range)])
        tiled_quantiles = original_tiled_quantiles * self.cos_basis_range * np.pi
        tiled_quantiles = torch.cos(tiled_quantiles)
        return self.phi(tiled_quantiles)

    def get_loss(self, embedded_obs, embedded_next_obs, batch_acts, batch_returns, dones_and_gamma, target_model=None):
        if not self.propagate_grad:
            embedded_obs = embedded_obs.detach()
            embedded_next_obs = embedded_next_obs.detach()

        if target_model is None:
            target_model = self

        batch_size = batch_acts.shape[0]
        online_current_quantile_estimates, tau = self.forward(embedded_obs,
                                                              n_quantile_samples=self.n_current_quantile_samples)

        tiled_batch_returns = torch.tile(batch_returns.view(-1, 1), [self.n_next_quantile_samples, 1])
        tiled_dones_and_gamma = torch.tile(dones_and_gamma.view(-1, 1), [self.n_next_quantile_samples, 1])

        with torch.no_grad():
            # No target model, bootstrap entirely from self.
            if target_model is self:
                online_next_quantile_estimates = self.forward(embedded_next_obs, n_quantile_samples=self.n_next_quantile_samples)[0]
                target_next_quantile_estimates = online_next_quantile_estimates

            # Double DQN
            elif self.use_double_q_learning:
                online_next_quantile_estimates = \
                    self.forward(embedded_next_obs, n_quantile_samples=self.n_next_quantile_samples)[0]
                target_next_quantile_estimates = \
                    target_model.forward(embedded_next_obs, n_quantile_samples=self.n_next_quantile_samples)[0]
            # DQN
            else:
                target_next_quantile_estimates = \
                    target_model.forward(embedded_next_obs, n_quantile_samples=self.n_next_quantile_samples)[0]
                online_next_quantile_estimates = target_next_quantile_estimates

            # (B, A)
            mean_next_online_values = online_next_quantile_estimates.view(self.n_next_quantile_samples,
                                                                          batch_size, -1).mean(dim=0)

            # (B, 1)
            best_next_actions = mean_next_online_values.argmax(dim=-1).view(-1, 1)

            # (B*T', 1)
            tiled_best_next_actions = torch.tile(best_next_actions, [self.n_next_quantile_samples, 1])

            batch_next_action_quantile_estimates = torch.gather(target_next_quantile_estimates, 1,
                                                                tiled_best_next_actions.view(-1, 1))

            if self.unsquish_function is not None:
                batch_next_action_quantile_estimates = self.unsquish_function(batch_next_action_quantile_estimates)

            # (B*T', 1)
            target_q = tiled_batch_returns + batch_next_action_quantile_estimates * tiled_dones_and_gamma

            if self.squish_function is not None:
                target_q = self.squish_function(target_q)

            # (T', B, 1)
            target_q = target_q.view(self.n_next_quantile_samples, batch_size, 1)

            # (B, T', 1)
            target_q = target_q.transpose(1, 0)

        # (B, 1)
        batch_actions = batch_acts.view(-1, 1)

        # (B*T, 1)
        tiled_batch_actions = torch.tile(batch_actions, [self.n_current_quantile_samples, 1])

        batch_action_quantile_estimates = torch.gather(online_current_quantile_estimates, dim=1,
                                                       index=tiled_batch_actions.view(-1, 1))
        # (T, B, 1)
        pred_q = batch_action_quantile_estimates.view(self.n_current_quantile_samples, batch_size, 1)

        # (B, T, 1)
        pred_q = pred_q.transpose(1, 0)

        # (B, T', T, 1)
        deltas = (target_q[:, :, None] - pred_q[:, None, :])

        # (B, T', T, 1)
        less_than_equal = (torch.le(deltas.abs(), self.huber_k)).float()
        greater_than = 1 - less_than_equal
        huber_loss_case_one = less_than_equal * 0.5 * deltas.square()
        huber_loss_case_two = greater_than * self.huber_k * (deltas.abs() - 0.5 * self.huber_k)
        huber_loss = huber_loss_case_one + huber_loss_case_two

        # (T, B, 1)
        replay_quantiles = tau.view(self.n_current_quantile_samples, batch_size, 1)

        # (B, T, 1)
        replay_quantiles = replay_quantiles.transpose(1, 0)

        # (B, T', T, 1)
        replay_quantiles = torch.tile(replay_quantiles[:, None, :, :],
                                      [1, self.n_next_quantile_samples, 1, 1]).float()

        deltas_less_zero = torch.where(deltas < 0, 1, 0).float().detach()

        # (B, T', T, 1)
        quantile_huber_loss = (torch.abs(replay_quantiles - deltas_less_zero) * huber_loss) / self.huber_k

        # (B, T', 1)
        loss = quantile_huber_loss.sum(dim=2)

        # (B)
        quantile_loss = loss.mean(dim=1).view(-1)

        return quantile_loss * self.distributional_loss_weight

    def log(self, logger):
        pass
