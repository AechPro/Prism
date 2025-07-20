import torch.nn as nn
import torch
import numpy as np
from prism.agents.models import IQNModel


class CompositeModel(nn.Module):
    def __init__(self, embedding_model, distribution_model, q_function_model, device="cpu", use_cuda_graph=False):
        super().__init__()
        self.embedding_model = embedding_model
        self.distribution_model = distribution_model
        self.q_function_model = q_function_model
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        self._static_input = None
        self._static_return_distribution_estimate = None
        self._static_q_outputs = None
        self._forward_cuda_graph = None
        self.should_build_forward_cuda_graph = use_cuda_graph
        self.loggables = {}

    def forward(self, x, for_action=True):
        if type(x) is not torch.Tensor:
            if type(x) is not np.array:
                x = np.asarray(x, dtype=np.float32)
            x = torch.from_numpy(x).float().to(self.device)

        if for_action and self.use_cuda_graph:
            return self._forward_with_cuda_graph(x)
        return self._forward_without_cuda_graph(x)

    def _forward_with_cuda_graph(self, x):
        if self._forward_cuda_graph is None:
            if self.should_build_forward_cuda_graph:
                self._build_forward_cuda_graph(x)
            else:
                return self._forward_without_cuda_graph(x)

        self._static_input.copy_(x)
        self._forward_cuda_graph.replay()

        q_output = None
        distribution_output = None
        if self._static_return_distribution_estimate is not None:
            distribution_output = self._static_return_distribution_estimate.clone()
        if self._static_q_outputs is not None:
            q_output = self._static_q_outputs.clone()

        return q_output, distribution_output

    def _forward_without_cuda_graph(self, x, for_action=True):
        if self.embedding_model is not None:
            embedded_state = self.embedding_model(x)
        else:
            embedded_state = x

        if self.distribution_model is not None:
            return_distribution_estimate = self.distribution_model(embedded_state, for_action=for_action)
        else:
            return_distribution_estimate = None

        if self.q_function_model is not None:
            q_function_estimates = self.q_function_model(embedded_state)
        else:
            if for_action and type(self.distribution_model) is IQNModel:
                q_function_estimates = [return_distribution_estimate.mean(dim=0)]
            else:
                q_function_estimates = None

        return q_function_estimates, return_distribution_estimate

    @torch.no_grad()
    def _build_forward_cuda_graph(self, x):
        self._static_input = x.clone().detach().requires_grad_(False)
        self._static_q_outputs = []

        print("Composite model building CUDA graph")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                q_function_estimates, return_distribution_estimate = self._forward_without_cuda_graph(x, for_action=True)

        torch.cuda.current_stream().wait_stream(s)
        self._forward_cuda_graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._forward_cuda_graph):
            q_output, return_distribution_estimate = self._forward_without_cuda_graph(self._static_input, for_action=True)
            if q_output is not None:
                self._static_q_outputs = q_output
            if return_distribution_estimate is not None:
                self._static_return_distribution_estimate = return_distribution_estimate

    def get_losses(self, batch, target_model):
        batch_obs = batch["observation"]
        batch_next_obs = batch["next"]["observation"]
        if batch_obs.shape[1] == 1:
            batch_obs = batch_obs.squeeze(1)
            batch_next_obs = batch_next_obs.squeeze(1)

        batch_returns = batch["next"]["reward"].flatten()
        batch_dones = batch["nonterminal"].flatten().float()
        batch_gammas = batch["gamma"].flatten().float()

        if len(batch["action"].shape) == 2 and batch["action"].shape[-1] != 1:
            batch_acts = batch["action"].argmax(dim=-1).flatten().long()
        else:
            batch_acts = batch["action"].flatten().long()

        dones_and_gamma = batch_gammas * batch_dones

        embedded_obs = self.embedding_model(batch_obs)
        with torch.no_grad():
            if target_model is not None:
                embedded_next_obs = target_model.embedding_model(batch_next_obs)
                distributional_target_model = target_model.distribution_model
                q_target_model = target_model.q_function_model
            else:
                embedded_next_obs = self.embedding_model(batch_next_obs)
                distributional_target_model = None
                q_target_model = None

        distribution_loss, q_loss, td_errors = None, None, None
        if self.distribution_model is not None:
            distribution_loss = self.distribution_model.get_loss(embedded_obs, embedded_next_obs, batch_acts,
                                                                 batch_returns,
                                                                 dones_and_gamma,
                                                                 target_model=distributional_target_model)

        if self.q_function_model is not None:
            q_loss = self.q_function_model.get_loss(embedded_obs, embedded_next_obs,
                                                    batch_acts, batch_returns, dones_and_gamma,
                                                    target_model=q_target_model)

        if distribution_loss is not None and q_loss is not None:
            td_errors = distribution_loss.detach() * 0.5 + q_loss.detach() * 0.5

        elif distribution_loss is not None:
            td_errors = distribution_loss.detach()

        elif q_loss is not None:
            td_errors = q_loss.abs().detach()

        return distribution_loss, q_loss, td_errors

    def log(self, logger):
        if self.embedding_model is not None:
            self.embedding_model.log(logger)

        if self.distribution_model is not None:
            self.distribution_model.log(logger)

        if self.q_function_model is not None:
            self.q_function_model.log(logger)
