import torch
import os
import pickle
import numpy as np


class Agent(object):
    def __init__(self, model, action_selector, eval_action_selector, optimizer, target_model,
                 use_cuda_graph, max_grad_norm):
        super().__init__()
        self.model = model
        self.target_model = target_model
        self.action_selector = action_selector
        self.eval_action_selector = eval_action_selector
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.use_cuda_graph = use_cuda_graph
        self.n_updates = 0
        self._is_eval = False

        self._static_per_weights = None
        self._static_distribution_loss = None
        self._static_q_loss = None
        self._static_total_loss = None
        self._static_new_per_weights = None
        self._static_batch = None
        self._learn_cuda_graph = None

        self.model.train()

    @torch.no_grad()
    def forward(self, obs):
        q_function_estimate, distribution_output = self.model(obs, for_action=True)
        if self._is_eval:
            action_probs = self.eval_action_selector.generate_action_probs(distribution_output, q_function_estimate)
            action = self.eval_action_selector.select_action(action_probs)
        else:
            action_probs = self.action_selector.generate_action_probs(distribution_output, q_function_estimate)
            action = self.action_selector.select_action(action_probs)

        return action

    def update(self, batch, per_weights=1):
        self.train()
        if self.use_cuda_graph:
            new_per_weights = self._update_with_cuda_graph(batch, per_weights)
        else:
            new_per_weights = self._update_without_cuda_graph(batch, per_weights)

        self.n_updates += 1
        return new_per_weights

    def _update_without_cuda_graph(self, batch, per_weights=1):
        distribution_loss, q_loss, td_errors = self.model.get_losses(batch, self.target_model)
        total_loss = 0
        new_per_weights = 0

        if distribution_loss is not None:
            total_loss += (distribution_loss * per_weights).mean()
            self._static_distribution_loss = distribution_loss

        if q_loss is not None:
            total_loss += (q_loss * per_weights).mean()
            self._static_q_loss = q_loss

        if td_errors is not None:
            new_per_weights = td_errors

        if type(total_loss) is torch.Tensor:
            self._static_total_loss = total_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        return new_per_weights

    def _update_with_cuda_graph(self, batch, per_weights=1):
        if self._learn_cuda_graph is None:
            self._build_update_cuda_graph(batch, per_weights)

            self._static_batch["observation"].copy_(batch["observation"])
            self._static_batch["next"]["observation"].copy_(batch["next"]["observation"])
            self._static_batch["next"]["reward"].copy_(batch["next"]["reward"])
            self._static_batch["action"].copy_(batch["action"])
            self._static_batch["nonterminal"].copy_(batch["nonterminal"])
            self._static_batch["gamma"].copy_(batch["gamma"])

        if type(per_weights) is torch.Tensor:
            self._static_per_weights.copy_(per_weights)
        self._learn_cuda_graph.replay()

        new_per_weights = per_weights
        if self._static_new_per_weights is not None:
            new_per_weights = self._static_new_per_weights.detach()

        return new_per_weights

    def _build_update_cuda_graph(self, batch, per_weights=1):
        self._static_batch = batch
        if type(per_weights) is not torch.Tensor:
            self._static_per_weights = torch.tensor(per_weights, device=self._static_batch.device)
        else:
            self._static_per_weights = per_weights.clone()

        print("Building Learn CUDA graph")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        total_loss = 0
        with torch.cuda.stream(s):
            for i in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                dl, ql, tde = self.model.get_losses(self._static_batch, self.target_model)
                if dl is not None:
                    total_loss += (dl * per_weights).mean()

                if ql is not None:
                    total_loss += (ql * per_weights).mean()

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss = 0
        torch.cuda.current_stream().wait_stream(s)

        self._learn_cuda_graph = torch.cuda.CUDAGraph()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0
        with torch.cuda.graph(self._learn_cuda_graph):
            self._static_distribution_loss, self._static_q_loss, self._static_new_per_weights = self.model.get_losses(
                self._static_batch, self.model)
            if self._static_distribution_loss is not None:
                total_loss += (self._static_distribution_loss * self._static_per_weights).mean()

            if self._static_q_loss is not None:
                total_loss += (self._static_q_loss * self._static_per_weights).mean()

            self._static_total_loss = total_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

    @torch.no_grad()
    def sync_target_model(self):
        for p1, p2 in zip(self.model.parameters(), self.target_model.parameters()):
            p2.data.copy_(p1.data)

    def save(self, directory):
        checkpoint_path = os.path.join(directory, 'agent')
        os.makedirs(checkpoint_path, exist_ok=True)

        # Save the model and optimizer state
        model_path = os.path.join(checkpoint_path, 'model.pt')
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.optimizer.state_dict(), optimizer_path)

        if self.target_model is not None:
            # Save the target model state
            target_model_path = os.path.join(checkpoint_path, 'target_model.pt')
            torch.save(self.target_model.state_dict(), target_model_path)

        # Save other stateful data
        state = {
            'action_selector': self.action_selector,
            'n_updates': self.n_updates,
            'eval_action_selector': self.eval_action_selector,
            'max_grad_norm': self.max_grad_norm,
            'use_cuda_graph': self.use_cuda_graph
        }
        with open(os.path.join(checkpoint_path, 'state.pkl'), 'wb') as f:
            pickle.dump(state, f)

    def load(self, directory):
        checkpoint_path = os.path.join(directory, 'agent')

        # Load the model and optimizer state
        model_path = os.path.join(checkpoint_path, 'model.pt')
        optimizer_path = os.path.join(checkpoint_path, 'optimizer.pt')
        self.model.load_state_dict(torch.load(model_path, map_location=self.model.device))
        self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.model.device))

        if self.target_model is not None:
            # Load the target model state
            target_model_path = os.path.join(checkpoint_path, 'target_model.pt')
            self.target_model.load_state_dict(
                torch.load(target_model_path, map_location=self.model.device))

        # Load other stateful data
        state_path = os.path.join(checkpoint_path, 'state.pkl')
        with open(state_path, 'rb') as f:
            state = pickle.load(f)

        self.action_selector = state['action_selector']
        self.eval_action_selector = state['eval_action_selector']
        self.max_grad_norm = state['max_grad_norm']
        self.use_cuda_graph = state['use_cuda_graph']
        self.n_updates = state['n_updates']

        self.train()

    def eval(self):
        self.model.eval()
        self._is_eval = True

    def train(self):
        self.model.train()
        self._is_eval = False

    @torch.no_grad()
    def log(self, logger):
        logger.log_data(data=self._static_total_loss.detach().item(),
                        group_name="Report/Losses", var_name="Total Loss")

        if self._static_distribution_loss is not None:
            logger.log_data(data=self._static_distribution_loss.detach().mean().item(),
                            group_name="Report/Losses", var_name="Distribution Loss")

        if self._static_q_loss is not None:
            logger.log_data(data=self._static_q_loss.detach().mean().item(),
                            group_name="Report/Losses", var_name="Q Loss")

        if logger.holdout_data is not None:
            idx = np.random.randint(0, logger.holdout_data["observation"].shape[0])
            obs = logger.holdout_data["observation"][idx]
            if obs.shape[0] != 1:
                obs = obs.unsqueeze(0)

            q_function_estimate, distribution_output = self.model._forward_without_cuda_graph(obs)
            self.action_selector.log(logger, distribution_output, q_function_estimate)

        self.model.log(logger)

