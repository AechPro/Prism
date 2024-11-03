def build_collector(config):
    from multiprocessing_experience_collection import MultiprocessExperienceCollector, ExperienceCollector

    inference_buffer_size = int(round(config.num_processes * 0.9))
    collector = ExperienceCollector(
        num_procs=config.num_processes,
        num_floats_per_process=config.shared_memory_num_floats_per_process,
        config=config,
        inference_buffer_size=inference_buffer_size,
        obs_stack_size=config.frame_stack_size,
        device=config.device,
        training_reward_ema=config.training_reward_ema
    )

    return collector
