

def build_algorithm(config):
    from prism.factory import collector_factory
    collector = collector_factory.build_collector(config)
    obs_shape, n_acts, n_agents = collector.get_env_info()

    import torch
    import os
    import numpy as np
    import random
    from prism.factory import agent_factory, exp_buffer_factory
    from prism.util.checkpointer import Checkpointer
    from prism.util import Logger

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    agent = agent_factory.build_agent(config, obs_shape, n_acts)

    agent.model.should_build_forward_cuda_graph = False
    collector.signal_processes_start_collecting(agent)
    agent.model.should_build_forward_cuda_graph = config.use_cuda_graph and "cuda" in config.device

    exp_buffer = exp_buffer_factory.build_exp_buffer(config)

    checkpoint_dir = os.path.join(config.checkpoint_dir, config.env_name)
    checkpointer = Checkpointer(checkpoint_dir, agent, exp_buffer,
                                config.timesteps_between_evaluations,
                                config.hours_per_checkpoint)

    logger = Logger(config, None)

    return agent, collector, exp_buffer, logger, checkpointer


def test():
    from prism.util.annealing_strategies import LinearAnneal
    import time
    import numpy as np
    from prism.config import MINATAR_CONFIG, SIMPLE_TRAP_CONFIG
    # import cProfile
    # prof = cProfile.Profile()

    # cfg = SIMPLE_TRAP_CONFIG
    cfg = MINATAR_CONFIG
    # cfg = LUNAR_LANDER_CFG
    # cfg = DEFAULT_CONFIG

    cfg.log_to_wandb = False
    cfg.render = True

    agent, collector, exp_buffer, logger, checkpointer = build_algorithm(cfg)

    # checkpointer = None
    # checkpointer.load_checkpoint("data/checkpoints/backup_checkpoint")

    print("Collecting initial timesteps...")
    total_ts = collector.collect_timesteps(cfg.num_initial_random_timesteps, agent, exp_buffer, random=True)

    holdout_data = exp_buffer.sample().clone()
    per_beta = LinearAnneal(cfg.per_beta_start, cfg.per_beta_end, cfg.per_beta_anneal_timesteps)

    ts_since_report = np.inf
    try:
        overall_steps_per_second_timer = time.time()
        current_ts = total_ts
        overall_steps_per_second = 0
        collected_steps_per_second = 0
        total_model_updates = 0

        ts_collection_time = 0
        batch_sampling_time = 0
        agent_update_time = 0
        update_priority_time = 0

        n_iters_since_measurement = 0
        avg_agent_update_time = 0
        avg_batch_sampling_time = 0
        avg_priority_update_time = 0

        steps_since_target_model_update = 0
        # prof.enable()

        while total_ts < cfg.timestep_limit:
            t1 = time.perf_counter()

            ts_this_iter = collector.collect_timesteps(cfg.timesteps_per_iteration, agent, exp_buffer, random=False)
            col_time_this_iter = time.perf_counter() - t1
            ts_collection_time += col_time_this_iter
            # print("Num timesteps:", ts_this_iter)
            # print("Collection time:", col_time_this_iter)

            total_ts += ts_this_iter
            ts_since_report += ts_this_iter
            steps_since_target_model_update += ts_this_iter

            t1 = time.perf_counter()
            batch, info = exp_buffer.sample(return_info=True)
            batch_sampling_time_this_iter = time.perf_counter() - t1
            batch_sampling_time += batch_sampling_time_this_iter
            # print("Batch sampling time:", batch_sampling_time_this_iter)

            if cfg.use_per:
                beta = per_beta.update(ts_this_iter)
                per_weights = info['_weight'].to(cfg.device)
                exp_buffer.buffer._sampler._beta = beta
            else:
                per_weights = 1

            t1 = time.perf_counter()
            new_per_weights = agent.update(batch, per_weights=per_weights)
            update_time_this_iter = time.perf_counter() - t1
            agent_update_time += update_time_this_iter

            if agent._static_batch is not None and total_model_updates == 0:
                exp_buffer._batch = agent._static_batch
            total_model_updates += 1
            # print("Agent update time:", batch_sampling_time_this_iter)
            # print()

            if cfg.use_double_q_learning and steps_since_target_model_update >= cfg.double_q_learning_target_update_period:
                agent.sync_target_model()
                steps_since_target_model_update = 0

            if cfg.use_per:
                t1 = time.perf_counter()
                exp_buffer.update_priority(info['index'], new_per_weights)
                update_priority_time += time.perf_counter() - t1

            if checkpointer is not None:
                checkpointer.checkpoint(total_ts)

            n_iters_since_measurement += 1
            if ts_since_report >= cfg.timesteps_per_report:

                time_since_sps_measure = time.time() - overall_steps_per_second_timer
                if time_since_sps_measure > 1:
                    ts_since_report = 0

                    overall_steps_per_second = (total_ts - current_ts) / time_since_sps_measure
                    collected_steps_per_second = (total_ts - current_ts) / ts_collection_time

                    avg_agent_update_time = agent_update_time / n_iters_since_measurement
                    avg_batch_sampling_time = batch_sampling_time / n_iters_since_measurement
                    avg_priority_update_time = update_priority_time / n_iters_since_measurement
                    n_iters_since_measurement = 0

                    batch_sampling_time = 0
                    agent_update_time = 0
                    update_priority_time = 0

                    current_ts = total_ts
                    overall_steps_per_second_timer = time.time()
                    ts_collection_time = 0

                    agent.log(logger)
                    collector.log(logger)


                    logger.report()
                    # prof.disable()
                    # prof.dump_stats("data/full_profile.pstat")
                    # prof.enable()
    finally:
        collector.close()
        exp_buffer.empty()


def test_wandb():
    import wandb
    wandb_run = wandb.init(project="rainbow-v2", group="debug", name="testing_features", reinit=True)
    wandb_run.log({"a": 1, "b": 2, "c": 3}, commit=False)
    for key in wandb_run.summary.keys():
        print("{}: {}".format(key, wandb_run.summary["key"]))


if __name__ == "__main__":
    test()
    # test_wandb()
