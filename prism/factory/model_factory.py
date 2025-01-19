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
                                    sparse_init_p=config.sparse_init_p,
                                    act_fn=act_fn,
                                    device=config.device)

    elif config.embedding_model_type == "nature_atari_cnn":
        embedding_model = NatureAtariCnn(frame_stack=config.frame_stack_size,
                                         feature_dim=config.embedding_model_final_dim,
                                         act_fn=act_fn,
                                         device=config.device,
                                         sparse_init_p=config.sparse_init_p,
                                         use_layer_norm=config.use_layer_norm)
        config.embedding_model_final_dim = embedding_model.output_dim

    elif config.embedding_model_type == "minatar_cnn":
        embedding_model = MinAtarModel(in_channels=env_state_shape[-1],
                                       act_fn=act_fn,
                                       device=config.device,
                                       sparse_init_p=config.sparse_init_p,
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
                             sparse_init_p=config.sparse_init_p,

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
                            sparse_init_p=config.sparse_init_p,
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
                            sparse_init_p=config.sparse_init_p,
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
