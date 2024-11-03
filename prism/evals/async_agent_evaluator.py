import multiprocessing as mp
import numpy as np


def eval_agent(agent, config, in_queue, out_queue):
    from prism.factory import env_factory
    from prism.util.observation_stacker import ObservationStacker
    import time
    import torch

    env = None
    try:
        config.use_cuda_graph = False
        obs_stacker = ObservationStacker(frame_stack=config.frame_stack_size, device=config.device)

        n_steps_to_eval = config.evaluation_timestep_horizon

        env = env_factory.build_environment(config)
        agent.eval()
        running = True

        while running:
            instructions = in_queue.get(block=True)
            if instructions is not None:
                if instructions[0] == "close":
                    running = False

                elif instructions[0] == "update":
                    agent.model.load_state_dict(instructions[1])

                elif instructions[0] == "eval":
                    print("Evaluating agent")
                    timesteps_marker = instructions[1]
                    obs, info = env.reset()
                    obs = torch.as_tensor(obs, dtype=torch.float32, device=config.device)
                    obs_stacker.reset(obs)

                    ep_rews = []
                    current_ep_rew = 0
                    for _ in range(n_steps_to_eval):
                        action = agent.forward(obs_stacker.obs_stack)
                        obs, rew, done, trunc, info = env.step(action.item())
                        obs = torch.as_tensor(obs, dtype=torch.float32, device=config.device)
                        obs_stacker.stack(obs)

                        current_ep_rew += rew
                        if done or trunc:
                            ep_rews.append(current_ep_rew)
                            current_ep_rew = 0
                            obs, info = env.reset()
                            obs = torch.as_tensor(obs, dtype=torch.float32, device=config.device)
                            obs_stacker.reset(obs)

                    out_queue.put(("eval_results", ep_rews, timesteps_marker))
                else:
                    time.sleep(1)

    finally:
        if env is not None:
            env.close()


class AsyncAgentEvaluator(object):
    def __init__(self, agent, config):
        self.agent = agent
        self.config = config

        can_fork = "forkserver" in mp.get_all_start_methods()
        start_method = "forkserver" if can_fork else "spawn"
        context = mp.get_context(start_method)

        self.in_queue = context.Queue()
        self.out_queue = context.Queue()
        self.process = context.Process(target=eval_agent, args=(agent, config, self.in_queue, self.out_queue))
        self.process.start()
        self.timesteps_since_last_eval = np.inf
        self.timesteps_between_evaluations = config.timesteps_between_evaluations
        self.waiting_queue_size = 0
        self.max_waiting_queue_size = 2

    def maybe_evaluate_agent(self, ts_this_iter, cumulative_timesteps):
        self.timesteps_since_last_eval += ts_this_iter
        if self.timesteps_since_last_eval >= self.timesteps_between_evaluations:
            self.timesteps_since_last_eval = 0
            return self.evaluate_agent(cumulative_timesteps)
        return None

    def evaluate_agent(self, cumulative_timesteps):
        self.in_queue.put(("update", self.agent.model.state_dict()))
        self.in_queue.put(("eval", cumulative_timesteps))
        self.waiting_queue_size += 1

        should_block = self.waiting_queue_size > self.max_waiting_queue_size
        if not self.out_queue.empty() or should_block:
            if should_block:
                print("EVAL AGENT QUEUE FULL - BLOCKING UNTIL RESULTS COLLECTED")
            data = self.out_queue.get(block=should_block)

            if data[0] == "eval_results":
                self.waiting_queue_size -= 1
                return data[1:]

        return None

    def close(self):
        self.in_queue.put(("close", None))
        self.process.join()
        self.in_queue.close()
        self.out_queue.close()


def test():
    from prism.factory import agent_factory, env_factory
    from prism.config import LUNAR_LANDER_CFG
    import time
    config = LUNAR_LANDER_CFG
    logger = None

    env = env_factory.build_environment(config)
    obs_shape = env.observation_space.shape
    n_acts = env.action_space.n
    one_hot = False

    agent = agent_factory.build_agent(config, logger, obs_shape, n_acts, one_hot)
    evaluator = AsyncAgentEvaluator(agent, config)
    for i in range(100):
        print(i, "|", evaluator.maybe_evaluate_agent(10_000, 10_000*i))
        time.sleep(1)
    evaluator.close()


if __name__ == "__main__":
    test()
