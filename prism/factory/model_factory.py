import torch

from prism.config import Config
from prism.agents.models import IQNModel, FFNNModel, QEnsemble, CompositeModel, NatureAtariCnn, MinAtarModel
from prism.agents import squish_functions
import torch.nn as nn
import numpy as np


def _parse_risk_policy(risk_policy_id):
    if risk_policy_id == "neutral" or risk_policy_id is None:
        return None
    return None


def parse_squish_function(squish_function_id):
    if squish_function_id == "obs_look_further":
        return squish_functions.obs_look_further_squish_fn, squish_functions.obs_look_further_squish_fn_inverse

    elif squish_function_id == "symlog":
        return squish_functions.symlog, squish_functions.symexp

    return None, None


def _parse_act_function(act_fn_id):
    if act_fn_id == "tanh":
        return nn.Tanh
    elif act_fn_id == "gelu":
        return nn.GELU
    elif act_fn_id == "sigmoid":
        return nn.Sigmoid
    elif act_fn_id == "swish":
        return nn.SiLU

    return nn.ReLU


def _parse_q_loss_function(q_loss_fn_id):
    if q_loss_fn_id == "mse":
        return nn.MSELoss(reduction='none')

    elif q_loss_fn_id == "huber":
        nn.SmoothL1Loss(reduction='none')

    return nn.MSELoss(reduction='none')  # Fallback to MSE.


def create_model(env_state_shape, env_n_actions, config: Config):
    act_fn = _parse_act_function(config.embedding_model_act_fn_id)
    squish_function, unsquish_function = parse_squish_function(config.loss_squish_fn_id)
    use_cuda_graph = config.use_cuda_graph and "cuda" in config.device

    embedding_model = None
    if "ffnn" in config.embedding_model_type:
        n_input_features = env_state_shape[-1]
        embedding_model = FFNNModel(n_input_features=n_input_features,
                                    n_output_features=config.embedding_model_final_dim,
                                    n_layers=config.embedding_model_num_layers,
                                    layer_width=config.embedding_model_layer_sizes,
                                    use_layer_norm=config.use_layer_norm,
                                    apply_layer_norm_first_layer=False,  # Don't normalize the input layer.
                                    output_act_fn=act_fn,
                                    act_fn=act_fn,
                                    device=config.device)

    elif config.embedding_model_type == "nature_atari_cnn":
        embedding_model = NatureAtariCnn(frame_stack=config.frame_stack_size,
                                         feature_dim=config.embedding_model_final_dim,
                                         act_fn=act_fn,
                                         device=config.device,
                                         use_layer_norm=config.use_layer_norm)
        config.embedding_model_final_dim = embedding_model.output_dim

    elif config.embedding_model_type == "minatar_cnn":
        embedding_model = MinAtarModel(in_channels=env_state_shape[-1],
                                       channels_first=False,
                                       act_fn=act_fn,
                                       device=config.device,
                                       use_layer_norm=config.use_layer_norm)
        config.embedding_model_final_dim = embedding_model.output_dim

    iqn_model = None
    if config.use_iqn:
        propagate_gradients = (config.ids_allow_distributional_gradients and config.use_ids) or not config.use_ids
        risk_policy = _parse_risk_policy(config.iqn_risk_policy_id)

        iqn_model = IQNModel(n_input_features=config.embedding_model_final_dim,
                             n_actions=env_n_actions,
                             n_basis_elements=config.iqn_n_basis_elements,
                             use_layer_norm=config.use_layer_norm,
                             n_model_layers=config.iqn_quantile_model_layers,
                             model_layer_size=config.iqn_quantile_model_feature_dim,
                             model_activation=act_fn,
                             use_double_q_learning=config.use_double_q_learning,

                             squish_function=squish_function,
                             unsquish_function=unsquish_function,
                             huber_k=config.iqn_huber_loss_kappa,
                             distributional_loss_weight=config.distributional_loss_weight,

                             n_current_quantile_samples=config.iqn_n_current_state_quantile_samples,
                             n_next_quantile_samples=config.iqn_n_next_state_quantile_samples,
                             n_quantile_samples_per_action=config.iqn_quantile_samples_per_action,
                             propagate_grad=propagate_gradients,
                             risk_policy=risk_policy,
                             device=config.device)

    q_model = None
    q_loss_function = _parse_q_loss_function(config.q_loss_fn)
    if config.use_ids:
        q_model = QEnsemble(n_input_features=config.embedding_model_final_dim,
                            n_actions=env_n_actions,
                            n_heads=config.ids_n_q_heads,
                            use_layer_norm=config.use_layer_norm,
                            n_model_layers=config.ids_n_q_head_model_layers,
                            model_layer_size=config.ids_q_head_feature_dim,
                            model_activation=act_fn,
                            squish_function=squish_function,
                            unsquish_function=unsquish_function,
                            use_double_q_learning=config.use_double_q_learning,
                            q_loss_function=q_loss_function,
                            q_loss_weight=config.q_loss_weight,
                            ensemble_variation_coef=config.ids_ensemble_variation_coef,
                            device=config.device)
    elif config.use_dqn:
        q_model = QEnsemble(n_input_features=config.embedding_model_final_dim,
                            n_actions=env_n_actions,
                            n_heads=1,
                            use_layer_norm=config.use_layer_norm,
                            n_model_layers=config.dqn_n_model_layers,
                            model_layer_size=config.dqn_n_model_feature_dim,
                            model_activation=act_fn,
                            squish_function=squish_function,
                            unsquish_function=unsquish_function,
                            use_double_q_learning=config.use_double_q_learning,
                            q_loss_function=q_loss_function,
                            q_loss_weight=config.q_loss_weight,
                            ensemble_variation_coef=0,
                            device=config.device)

    composite_model = CompositeModel(embedding_model=embedding_model, distribution_model=iqn_model,
                                     q_function_model=q_model, device=config.device,
                                     use_cuda_graph=use_cuda_graph)

    print("Built model with {} parameters:".format(torch.nn.utils.parameters_to_vector(composite_model.parameters()).numel()))
    print(composite_model)
    return composite_model


def test():
    import torch
    from prism.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    cfg.embedding_model_type = "basic_ffnn"
    cfg.embedding_model_num_layers = 6
    cfg.embedding_model_layer_sizes = 3000
    cfg.embedding_model_final_dim = 256
    cfg.iqn_quantile_model_layers = 3
    cfg.iqn_quantile_model_feature_dim = 512
    cfg.ids_n_q_head_model_layers = 4
    cfg.ids_q_head_feature_dim = 256
    cfg.embedding_model_act_fn_id = "gelu"
    cfg.iqn_n_basis_elements = 128
    cfg.iqn_n_current_state_quantile_samples = 16
    cfg.iqn_n_next_state_quantile_samples = 16
    cfg.learning_rate = 5e-5
    cfg.ids_n_q_heads = 5
    cfg.use_ids = True
    cfg.use_iqn = True
    cfg.use_layer_norm = False
    cfg.device = "cpu"

    torch.manual_seed(123)
    np.random.seed(123)

    model = create_model(env_state_shape=(27,),
                         env_n_actions=11,
                         config=cfg)

    print(model)

    dummy_data = torch.ones(1, 27, device=cfg.device)
    q_head_outputs, distribution_model_output = model(dummy_data, for_action=True)

    if distribution_model_output is not None:
        distribution = distribution_model_output.mean(dim=0)

        print("distribution:")
        print(distribution)

    if q_head_outputs is not None:
        print("q_head_outputs:")
        print(q_head_outputs)

    loss = distribution.sum()
    for out in q_head_outputs:
        loss += out.sum()

    print("LOSS", loss.item())
    loss.backward()

    with torch.no_grad():
        grad_vec = []
        for param in model.parameters():
            grad_vec.append(param.grad.view(-1))
            print(param.shape, "|", param.grad.norm())
        grad_vec = torch.cat(grad_vec, dim=0)
        print("GRAD NORM", grad_vec.norm())


def test_losses():
    from tensordict import TensorDict
    from prism.factory import agent_factory
    from prism.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    cfg.embedding_model_type = "basic_ffnn"
    cfg.embedding_model_num_layers = 6
    cfg.embedding_model_layer_sizes = 3000
    cfg.embedding_model_final_dim = 256
    cfg.iqn_quantile_model_layers = 3
    cfg.iqn_quantile_model_feature_dim = 512
    cfg.ids_n_q_head_model_layers = 4
    cfg.ids_q_head_feature_dim = 256
    cfg.embedding_model_act_fn_id = "gelu"
    cfg.iqn_n_basis_elements = 128
    cfg.iqn_n_current_state_quantile_samples = 16
    cfg.iqn_n_next_state_quantile_samples = 16
    cfg.learning_rate = 5e-5
    cfg.ids_n_q_heads = 5
    cfg.use_ids = True
    cfg.use_iqn = True
    cfg.use_layer_norm = True
    cfg.device = "cpu"

    torch.manual_seed(123)
    np.random.seed(123)

    bs = 32
    model = create_model(env_state_shape=(27,),
                         env_n_actions=11,
                         config=cfg)

    fake_batch = TensorDict({
        "observation": torch.ones(bs, 27, device=cfg.device),
        "next": {"observation": torch.ones(bs, 27, device=cfg.device),
                 "reward": torch.ones(bs, 1, device=cfg.device)},
        "action": torch.ones(bs, 1, device=cfg.device),
        "nonterminal": torch.ones(bs, 1, device=cfg.device),

        "gamma": torch.ones(bs, 1, device=cfg.device),
    })
    distribution_loss, q_loss, td_errors = model.get_losses(fake_batch, None)
    total_loss = distribution_loss.mean() + q_loss.mean() / len(model.q_function_model.q_heads)
    total_loss.backward()

    with torch.no_grad():
        grad_vec = []
        for param in model.parameters():
            grad_vec.append(param.grad.view(-1))
            print(param.shape, "|", param.grad.norm())
        grad_vec = torch.cat(grad_vec, dim=0)
        print("GRAD NORM", grad_vec.norm())


def test_cuda_graph():
    import time
    from tensordict import TensorDict
    from prism.config import DEFAULT_CONFIG

    cfg = DEFAULT_CONFIG
    model = create_model(env_state_shape=(84, 84, 4), env_n_actions=10, config=cfg)

    # fake_input = torch.ones(16, 84, 84, 4, device=cfg.device)
    # graph_callable = torch.cuda.make_graphed_callables(model, (fake_input,))
    # real_input = torch.rand(16, 84, 84, 4, device=cfg.device)

    # print("FORWARD PASS COMPARISON")
    # with torch.no_grad():
    #     t1 = time.perf_counter()
    #     for i in range(1000):
    #         out = graph_callable(real_input)
    #     t2 = time.perf_counter()
    #     print("Graph Callable", t2 - t1)
    #
    #     t1 = time.perf_counter()
    #     for i in range(1000):
    #         out = model(real_input)
    #     t2 = time.perf_counter()
    #     print("Regular PyTorch", t2 - t1)

    print("BACKWARD PASS COMPARISON")
    fake_batch = TensorDict({
        "observation": torch.ones(32, 84, 84, 4, device=cfg.device),
        "next": {"observation": torch.ones(32, 84, 84, 4, device=cfg.device),
                 "reward": torch.ones(32, 1, device=cfg.device)},
        "action": torch.ones(32, 1, device=cfg.device),
        "nonterminal": torch.ones(32, 1, device=cfg.device),
        "gamma": torch.ones(32, 1, device=cfg.device),
    })

    optim = torch.optim.Adam(model.parameters(), lr=5e-5, capturable=True)

    print("Building CUDA graph")
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(3):
            optim.zero_grad(set_to_none=True)
            dl, ql, tde = model.get_losses(fake_batch, model)
            total_loss = (dl * 1).mean() + (ql * 1).mean() / len(model.q_function_model.q_heads)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()
    torch.cuda.current_stream().wait_stream(s)

    s.synchronize()
    print("Capturing:", torch.cuda.is_current_stream_capturing())
    g = torch.cuda.CUDAGraph()
    optim.zero_grad(set_to_none=True)

    with torch.cuda.graph(g):
        distribution_loss, q_loss, td_errors = model.get_losses(fake_batch, model)
        total_loss = (distribution_loss * 1).mean() + (q_loss * 1).mean() / len(model.q_function_model.q_heads)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()

    print("Running graph replay")
    t1 = time.perf_counter()
    for i in range(100):
        fake_batch["observation"].copy_(torch.rand(32, 84, 84, 4, device=cfg.device))
        fake_batch["next"]["observation"].copy_(torch.rand(32, 84, 84, 4, device=cfg.device))
        fake_batch["next"]["reward"].copy_(torch.rand(32, 1, device=cfg.device))
        fake_batch["action"].copy_(torch.rand(32, 1, device=cfg.device))
        fake_batch["nonterminal"].copy_(torch.rand(32, 1, device=cfg.device))
        fake_batch["gamma"].copy_(torch.rand(32, 1, device=cfg.device))
        print(distribution_loss.detach().mean().item())
        g.replay()
    t2 = time.perf_counter()
    print("Graph replay", t2 - t1)

    print("Running regular PyTorch")
    t1 = time.perf_counter()
    for i in range(100):
        fake_batch["observation"].copy_(torch.rand(32, 84, 84, 4, device=cfg.device))
        fake_batch["next"]["observation"].copy_(torch.rand(32, 84, 84, 4, device=cfg.device))
        fake_batch["next"]["reward"].copy_(torch.rand(32, 1, device=cfg.device))
        fake_batch["action"].copy_(torch.rand(32, 1, device=cfg.device))
        fake_batch["nonterminal"].copy_(torch.rand(32, 1, device=cfg.device))
        fake_batch["gamma"].copy_(torch.rand(32, 1, device=cfg.device))
        optim.zero_grad()
        distribution_loss, q_loss, td_errors = model.get_losses(fake_batch, model)
        total_loss = (distribution_loss * 1).mean() + (q_loss * 1).mean() / len(model.q_function_model.q_heads)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
    t2 = time.perf_counter()
    print("Regular PyTorch", t2 - t1)


if __name__ == "__main__":
    # test()
    # test_losses()
    test_cuda_graph()
