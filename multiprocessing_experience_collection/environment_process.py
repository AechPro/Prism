

def run_env(proc_id, config, shared_memory, shared_memory_offset, local_memory_size):
    import sys
    import time
    import numpy as np
    from prism.factory import env_factory
    from multiprocessing_experience_collection import EnvProcessMemoryInterface

    # Check if torch is imported. We want this to be False.
    print("TORCH IMPORTED:", "torch" in sys.modules)
    timeout_seconds = 1200

    # Get our slice of the shared memory buffer.
    local_memory_slice = np.frombuffer(shared_memory, dtype=np.float32,
                                       offset=shared_memory_offset, count=local_memory_size)

    # Set up the memory interface.
    memory_interface = EnvProcessMemoryInterface(local_memory_slice)

    # Get current order, there's probably nothing here yet.
    current_order = memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX)
    env = None

    should_render = proc_id == 0 and config.render

    config.render = should_render
    config.seed += proc_id

    try:
        # Build the environment and get initial obs.
        env = env_factory.build_environment(config)
        obs, _ = env.reset()

        if "ALE" in config.env_name:
            expected_obs_length = 2
        elif "MinAtar" in config.env_name:
            expected_obs_length = 3
        else:
            expected_obs_length = 1

        obs_shape = np.shape(obs)
        if len(obs_shape) == expected_obs_length:
            n_current_agents = 1
        else:
            n_current_agents = obs_shape[0]
        # Wait for first order.
        while (current_order != EnvProcessMemoryInterface.PROC_ORDER_GET_ENV_INFO_FLAG and
               current_order != EnvProcessMemoryInterface.PROC_ORDER_START_FLAG and
               current_order != EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG):

            current_order = memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX)
            time.sleep(0.1)

        # Quit if halt is ordered.
        if current_order == EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG:
            return

        # If requested, write env info to shared memory and wait for reply.
        current_order = memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX)
        if current_order == EnvProcessMemoryInterface.PROC_ORDER_GET_ENV_INFO_FLAG:

            # Write env info..
            memory_interface.write_env_info(env.observation_space.shape, env.action_space.n, n_current_agents)

            # Wait for reply.
            while (current_order != EnvProcessMemoryInterface.PROC_ORDER_START_FLAG and
                   current_order != EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG):

                current_order = memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX)
                time.sleep(0.1)

        # Quit if halt is ordered.
        if current_order == EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG:
            return

        # Write initial observation and wait for first action.
        memory_interface.write_reset_obs(obs, n_current_agents)

        # Wait for start order.
        current_order = memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX)
        while (current_order != EnvProcessMemoryInterface.PROC_ORDER_START_FLAG and
               current_order != EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG):

            current_order = memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX)
            time.sleep(0.1)

        # Quit if halt is ordered.
        if current_order == EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG:
            return

        # Run collection loop.
        timeout_timer_start = time.time()

        while current_order != EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG:
            # Check if timer has expired. This is an emergency stop in case something happened and this process is
            # spinning with nothing connected to it.
            current_time = time.time()
            if current_time - timeout_timer_start > timeout_seconds:
                break

            # If we've been told to wait, spin until we get a new order.
            current_order = memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX)
            # If the collector told us to step, and we're waiting for an action, do it.
            if current_order == EnvProcessMemoryInterface.PROC_ORDER_TAKE_STEP_FLAG:
                render_timer = time.perf_counter()

                # Reset timeout timer.
                timeout_timer_start = current_time

                # Read the current action.
                action = memory_interface.read_action(n_current_agents)

                # Let the collector know we're stepping.
                memory_interface.set_flag(EnvProcessMemoryInterface.PROC_STATUS_IDX, EnvProcessMemoryInterface.PROC_STATUS_STEPPING_ENV_FLAG)

                # Have to check here just in case something happened at an inconvenient time.
                if memory_interface.get_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX) != EnvProcessMemoryInterface.PROC_ORDER_HALT_FLAG:

                    # Erase current order.
                    memory_interface.set_flag(EnvProcessMemoryInterface.PROC_ORDER_IDX, EnvProcessMemoryInterface.PROC_ORDER_NULL_FLAG)

                # Step the env.
                next_obs, rew, done, truncated, info = env.step(action)

                # If a reset is needed we send the stepped observation as the current obs and the reset observation
                # as the next obs. This is necessary because we also reset on truncated, and the next obs will be valid
                # in that case.
                if done or truncated:
                    obs = next_obs
                    next_obs, _ = env.reset()

                # Check the number of agents in the current obs and the number of agents in the next obs.
                current_obs_shape = np.shape(obs)
                next_obs_shape = np.shape(next_obs)

                if len(current_obs_shape) == expected_obs_length:
                    n_current_agents = 1
                else:
                    n_current_agents = current_obs_shape[0]

                if len(next_obs_shape) == expected_obs_length:
                    n_next_agents = 1
                else:
                    n_next_agents = next_obs_shape[0]

                # Write step data. This also sets our current status flag to waiting for an action.
                memory_interface.write_step_data(obs, rew, done, truncated, next_obs, n_current_agents, n_next_agents)
                obs = next_obs
                if config.render:
                    step_time = time.perf_counter() - render_timer
                    if step_time < 0.016667:
                        time.sleep(0.016667 - step_time)
            else:
                time.sleep(0.001)

    finally:
        # Close the environment if it exists.
        print("Environment process {} closing...".format(proc_id))
        if env is not None:
            env.close()

        # Tell the collector that we've crashed.
        memory_interface.set_flag(EnvProcessMemoryInterface.PROC_STATUS_IDX, EnvProcessMemoryInterface.PROC_CRASHED_FLAG)
        print("Environment process {} closed!".format(proc_id))
