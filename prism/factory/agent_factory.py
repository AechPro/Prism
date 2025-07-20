from prism.config import Config
from prism.agents import Agent, action_selectors
from prism.factory import model_factory
import torch


def build_agent(config: Config, obs_shape, n_actions):
    use_cuda_graph = config.use_cuda_graph and "cuda" in config.device

    obs_shape = [int(arg) for arg in obs_shape]
    n_actions = int(n_actions)
    model = model_factory.create_model(obs_shape, n_actions, config)
    eval_action_selector = action_selectors.GreedyActionSelector()

    target_model = None
    if config.use_target_network:
        target_model = model_factory.create_model(obs_shape, n_actions, config)
        target_model.load_state_dict(model.state_dict())

    if config.use_ids:
        squish, unsquish = model_factory.parse_squish_function(config.loss_squish_fn_id)
        action_selector = action_selectors.IDSActionSelector(config.ids_lambda,
                                                             config.ids_use_random_samples,
                                                             config.ids_epsilon,
                                                             config.ids_rho_lower_bound,
                                                             config.ids_beta,
                                                             unsquish)

        # eval_action_selector = action_selectors.MinRegretActionSelector(lmbda=config.ids_lambda,
        #                                                                 unsquish_function=unsquish)

    elif config.use_e_greedy:
        action_selector = action_selectors.EGreedyActionSelector(config.e_greedy_initial_epsilon,
                                                                 config.e_greedy_final_epsilon,
                                                                 config.e_greedy_decay_timesteps,
                                                                 config.seed)
    else:
        action_selector = action_selectors.GreedyActionSelector()

    if config.use_adam:
        # import ASGD
        # optimizer = ASGD.ASGD(model.parameters(), lr=config.learning_rate)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config.learning_rate,
                                     betas=(config.adam_beta1, config.adam_beta2),
                                     eps=config.adam_epsilon, capturable=use_cuda_graph)
    elif config.use_rmsprop:
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=config.learning_rate,
                                        alpha=config.rmsprop_alpha,
                                        centered=True,
                                        eps=config.rmsprop_epsilon,
                                        capturable=use_cuda_graph)

    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=config.learning_rate)

    return Agent(model, action_selector, eval_action_selector, optimizer,
                 target_model, use_cuda_graph, config.max_grad_norm)
