import torch


class ActionSelector(object):
    def __init__(self, logger=None):
        self.logger = logger
        self.loggables = {}

    def generate_action_probs(self, *args, **kwargs):
        raise NotImplementedError

    def select_action(self, action_probs):
        raise NotImplementedError

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, logger, action_value_distribution, q_estimates):
        pass


class EGreedyActionSelector(ActionSelector):
    def __init__(self, e_start, e_stop, anneal_time, seed=123):
        super().__init__()

        import numpy as np
        from prism.util.annealing_strategies import LinearAnneal
        self.epsilon = LinearAnneal(start_value=e_start, stop_value=e_stop, max_steps=anneal_time)
        self.greedy = GreedyActionSelector()
        self.rng = np.random.RandomState(seed)

    def generate_action_probs(self, action_value_distribution, q_estimates):
        n_timesteps, n_actions = q_estimates[0].shape
        if self.rng.uniform(0, 1) < self.epsilon.update(n_timesteps):
            action = self.rng.randint(n_actions, size=(n_timesteps,))
            action = torch.as_tensor(action, dtype=torch.long)
            action_probs = torch.nn.functional.one_hot(action, n_actions)
        else:
            action_probs = self.greedy.generate_action_probs(action_value_distribution, q_estimates)

        return action_probs

    def select_action(self, action_probs):
        return torch.argmax(action_probs, dim=-1)

    def save(self, path):
        import os
        eps_path = os.path.join(path, 'epsilon.txt')
        with open(eps_path, 'w') as f:
            f.write(str(self.epsilon.get_state()))
            f.write("\n")

    def load(self, path):
        import os
        eps_path = os.path.join(path, 'epsilon.txt')

        if os.path.exists(eps_path):
            with open(eps_path, 'r') as f:
                lines = f.readlines()
                state = int(lines[0])
                self.epsilon.set_state(state)

    def log(self, logger, action_value_distribution, q_estimates):
        logger.log_data(data=self.epsilon.get_value(), group_name="Report/Action Selector", var_name="Epsilon")


class GreedyActionSelector(ActionSelector):
    def generate_action_probs(self, action_value_distribution, q_estimates):
        mean = 0
        for q_estimate in q_estimates:
            mean += q_estimate
        mean /= len(q_estimates)
        action = torch.argmax(mean, dim=-1)

        action_probs = torch.nn.functional.one_hot(action, mean.shape[-1])
        return action_probs

        # if onehot_actions:
        #     action = torch.nn.functional.one_hot(action, mean.shape[-1])
        #
        # return action.long()

    def select_action(self, action_probs):
        return torch.argmax(action_probs, dim=-1).long().view(-1)

    def log(self, logger, action_value_distribution, q_estimates):
        pass


class MinRegretActionSelector(ActionSelector):
    def __init__(self, lmbda,  unsquish_function=None):
        super().__init__()
        self.lmbda = lmbda
        self.unsquish_function = unsquish_function

    def generate_action_probs(self, action_value_distribution, q_estimates):
        q_shape = q_estimates[0].shape
        unsquish = self.unsquish_function
        mean = 0
        for i in range(len(q_estimates)):
            if unsquish is not None:
                q_estimates[i] = unsquish(q_estimates[i])
            mean += q_estimates[i]
        mean /= len(q_estimates)

        variance = 0
        for q_estimate in q_estimates:
            variance += (q_estimate - mean).square()
        variance /= len(q_estimates)

        std = torch.sqrt(variance)

        regret = torch.max(mean + self.lmbda * std, dim=-1).values.view(-1, 1)
        regret = regret - (mean - self.lmbda * std)
        regret_sq = torch.square(regret).view(q_shape)
        action = torch.argmin(regret_sq, dim=-1)

        return torch.nn.functional.one_hot(action, regret_sq.shape[-1])

    def select_action(self, action_probs):
        action = action_probs.argmax(dim=-1)
        action = action.long().view(-1)
        return action


class IDSActionSelector(ActionSelector):
    def __init__(self, lmbda, random_sample, epsilon, ids_rho_lower_bound, beta, unsquish_function=None):
        super().__init__()
        self.random_sample = random_sample
        self.beta = beta
        self.lmbda = lmbda
        self.epsilon = epsilon
        self.ids_rho_lower_bound = ids_rho_lower_bound
        self.softmax = torch.nn.Softmax(dim=-1)
        self.unsquish_function = unsquish_function

    def generate_action_probs(self, action_value_distribution, q_estimates, for_log=False):
        q_shape = q_estimates[0].shape

        unsquish = self.unsquish_function
        if unsquish is not None:
            action_value_distribution = unsquish(action_value_distribution)

        mean = 0
        for i in range(len(q_estimates)):
            if unsquish is not None:
                q_estimates[i] = unsquish(q_estimates[i])
            mean += q_estimates[i]
        mean /= len(q_estimates)

        variance = 0
        for q_estimate in q_estimates:
            variance += (q_estimate - mean).square()
        variance /= len(q_estimates)

        std = torch.sqrt(variance)

        regret = torch.max(mean + self.lmbda * std, dim=-1).values.view(-1, 1)
        regret = regret - (mean - self.lmbda * std)
        regret_sq = torch.square(regret).view(q_shape)

        var_z = action_value_distribution.var(dim=0)
        normalized_var_z = var_z / (self.epsilon + var_z.mean(dim=-1).unsqueeze(-1))
        rho = torch.clamp(normalized_var_z, min=self.ids_rho_lower_bound)

        # info_gain = 1 + variance.view(q_shape) / rho
        info_gain = torch.log(1 + variance.view(q_shape) / rho) + self.epsilon
        # ids_scores = - self.beta * info_gain
        ids_scores = regret_sq / info_gain

        if self.random_sample:
            action_probs = self.softmax(-ids_scores)
            action_probs = torch.clamp(action_probs, min=self.epsilon, max=1)
        else:
            action = torch.argmin(ids_scores, dim=-1)
            action_probs = torch.nn.functional.one_hot(action, ids_scores.shape[-1])

        if for_log:
            self.loggables["Action Regret"] = regret
            self.loggables["Q Estimate Ensemble Mean"] = mean
            self.loggables["Q Estimate Ensemble Variance"] = variance
            self.loggables["Return Distribution Variance"] = var_z
            self.loggables["Information Gain"] = info_gain
            self.loggables["IDS Scores"] = ids_scores
            self.loggables["Action Probs"] = action_probs
            self.loggables["Normalized Return Distribution Variance"] = normalized_var_z

        return action_probs

    def select_action(self, action_probs):
        if not self.random_sample:
            action = action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(action_probs, 1, True)

        action = action.long().view(-1)
        return action

    def log(self, logger, action_value_distribution, q_estimates):
        self.generate_action_probs(action_value_distribution, q_estimates, for_log=True)
        for key, value in self.loggables.items():
            if isinstance(value, torch.Tensor):
                value = value.tolist()
                decimals_to_round = 4
                for i in range(len(value)):
                    for j in range(len(value[i])):
                        value[i][j] = round(value[i][j], decimals_to_round)
                if len(value) == 1:
                    value = value[0]

            logger.log_data(data=value,
                            group_name="Debug/IDS",
                            var_name=key)


class SimpleIDSActionSelector(ActionSelector):
    def __init__(self, lmbda, random_sample, epsilon, ids_rho_lower_bound, beta, unsquish_function=None):
        super().__init__()
        self.random_sample = random_sample
        self.lmbda = lmbda
        self.beta = beta
        self.epsilon = epsilon
        self.ids_rho_lower_bound = ids_rho_lower_bound
        self.softmax = torch.nn.Softmax(dim=-1)
        self.unsquish_function = unsquish_function

    def generate_action_probs(self, action_value_distribution, q_estimates, for_log=False):
        q_shape = q_estimates[0].shape

        unsquish = self.unsquish_function
        if unsquish is not None:
            action_value_distribution = unsquish(action_value_distribution)

        mean = 0
        for i in range(len(q_estimates)):
            if unsquish is not None:
                q_estimates[i] = unsquish(q_estimates[i])
            mean += q_estimates[i]
        mean /= len(q_estimates)

        # variance = 0
        # for q_estimate in q_estimates:
        #     variance += (q_estimate - mean).square()
        # variance /= len(q_estimates)

        # Compute the maximum distance between q estimates at each action:
        max_diff = torch.zeros_like(mean)
        for i in range(len(q_estimates)):
            for j in range(i + 1, len(q_estimates)):
                diff = torch.abs(q_estimates[i] - q_estimates[j])
                max_diff = torch.max(max_diff, diff)

        # std = torch.sqrt(variance)
        # regret = torch.max(mean + self.lmbda * std, dim=-1).values.view(-1, 1)
        # regret = regret - (mean - self.lmbda * std)
        # regret_sq = torch.square(regret).view(q_shape)
        #
        # var_z = action_value_distribution.var(dim=0)
        # normalized_var_z = var_z / (self.epsilon + var_z.mean(dim=-1).unsqueeze(-1))
        #
        # rho = torch.clamp(normalized_var_z, min=self.ids_rho_lower_bound)
        #
        # info_gain = torch.log(1 + variance.view(q_shape) / rho) + self.epsilon

        quantile_difference = (torch.max(action_value_distribution, dim=0).values -
                               torch.min(action_value_distribution, dim=0).values).abs()

        q_values = mean - mean.mean(dim=-1).view(-1, 1)
        rescaled_uncertainty = max_diff
        action_uncertainty = rescaled_uncertainty
        scaled_q_values = (1 - self.beta) * action_uncertainty + q_values * self.beta

        action_probs = torch.nn.functional.one_hot(torch.argmax(scaled_q_values, dim=-1), scaled_q_values.shape[-1])
        # action_probs = self.softmax(scaled_q_values)

        if for_log:
            self.loggables["Q Estimate Max Difference"] = max_diff
            self.loggables["Q Estimate Ensemble Mean"] = mean
            self.loggables["Action Uncertainty"] = action_uncertainty
            self.loggables["Scaled Q Values"] = scaled_q_values
            self.loggables["Action Probs"] = action_probs

        return action_probs

    def select_action(self, action_probs):
        if not self.random_sample:
            action = action_probs.argmax(dim=-1)
        else:
            action = torch.multinomial(action_probs, 1, True)

        action = action.long().view(-1)
        return action

    def log(self, logger, action_value_distribution, q_estimates):
        self.generate_action_probs(action_value_distribution, q_estimates, for_log=True)
        for key, value in self.loggables.items():
            if isinstance(value, torch.Tensor):
                value = value.tolist()
                decimals_to_round = 4
                for i in range(len(value)):
                    for j in range(len(value[i])):
                        value[i][j] = round(value[i][j], decimals_to_round)
                if len(value) == 1:
                    value = value[0]
            logger.log_data(data=value,
                            group_name="Debug/SimpleIDS",
                            var_name=key)

