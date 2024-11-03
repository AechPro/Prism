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


def test_cuda_graph():
    from prism.factory import env_factory
    from tensordict import TensorDict
    from prism.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    cfg.num_processes = 1
    env = env_factory.build_environment(cfg)

    agent = build_agent(env, cfg)

    for i in range(100):
        fake_per_weights = torch.rand(32, device=cfg.device)
        fake_batch = TensorDict({
            "observation": torch.rand(32, 84, 84, 4, device=cfg.device),
            "next": {"observation": torch.rand(32, 84, 84, 4, device=cfg.device),
                     "reward": torch.rand(32, 1, device=cfg.device)},
            "action": torch.randint(0, 3, (32, 1), device=cfg.device),
            "nonterminal": torch.zeros(32, 1, device=cfg.device),
            "gamma": torch.ones(32, 1, device=cfg.device),
        })

        distribution_loss_report, q_loss_report, new_per_weights, update_magnitude = agent.update(fake_batch,
                                                                                                  fake_per_weights)
        print(distribution_loss_report, q_loss_report, new_per_weights, update_magnitude)


def test_random_cuda_graph():
    print("Building CUDA graph")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            x = torch.rand(10, 10, device="cuda")
            y = x * x.abs().sum(dim=0)
            total_loss = y.mean()
    torch.cuda.current_stream().wait_stream(s)

    cuda_graph = torch.cuda.CUDAGraph()

    with torch.cuda.graph(cuda_graph):
        static_random_data = torch.rand(10, 10, device="cuda")
        y = x * x.abs().sum(dim=0)
        static_loss_value = y.mean()

    print("before", static_random_data)
    for i in range(10):
        cuda_graph.replay()
        print()
        print("after", static_random_data)


def test_agent_forward_cuda_graph():
    from prism.factory import env_factory
    from tensordict import TensorDict
    from prism.config import LUNAR_LANDER_CFG

    cfg = LUNAR_LANDER_CFG
    cfg.use_cuda_graph = True
    cfg.use_ids = False
    cfg.use_iqn = True

    agent = build_agent(cfg, None, (8,), 4, False)
    obs_batch = torch.ones(1, 8, dtype=torch.float32, device=cfg.device)
    prev = None
    for i in range(10):
        out = agent.forward(obs_batch)
        if prev is not None:
            print(out.tolist())
            print()
        prev = out


if __name__ == "__main__":
    test_agent_forward_cuda_graph()
    # test_random_cuda_graph()
    # test_cuda_graph()
