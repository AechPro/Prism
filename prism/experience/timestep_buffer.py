from tensordict import TensorDict
from prism.experience import Timestep
import torch
import weakref
import os
import pickle
import time


class TimestepBuffer(object):
    def __init__(self, torchrl_buffer, frame_stack=1, device="cpu", n_step=3, gamma=0.99):
        self.buffer = torchrl_buffer
        self.device = device
        self.frame_stack = frame_stack
        self.n_step = n_step
        self._batch = None
        self.gammas = [gamma ** i for i in range(n_step + 1)]  # need an extra gamma for the bootstrap equation

        self._obs = None
        self._next_obs = None
        self._reward = None
        self._nonterminal = None
        self._gamma = None
        self._action = None
        self._cpu_obs_buffer = None
        self._cpu_next_obs_buffer = None
        self._cpu_reward_buffer = None
        self._cpu_nonterminal_buffer = None
        self._cpu_gamma_buffer = None
        self._cpu_action_buffer = None

    def extend(self, timestep: Timestep):
        return self.buffer.extend([timestep])

    @torch.no_grad()
    def sample(self, batch_size=None, return_info=False):
        data = self.buffer.sample(batch_size=batch_size, return_info=return_info)
        if return_info:
            timesteps, info = data
        else:
            timesteps = data

        if batch_size is None:
            batch_size = self.buffer._batch_size

        td_batch = self._timesteps_to_batch(timesteps, batch_size)

        if return_info:
            return td_batch, info
        else:
            return td_batch

    def update_priority(self, indices, priorities):
        self.buffer.update_priority(indices, priorities)

    def set_static_batch(self, batch):
        self._batch = batch
        self._obs = self._batch["observation"]
        self._next_obs = self._batch["next"]["observation"]
        self._reward = self._batch["next"]["reward"]
        self._nonterminal = self._batch["nonterminal"]
        self._gamma = self._batch["gamma"]
        self._action = self._batch["action"]

        if "cpu" not in self.device:
            self._cpu_obs_buffer = torch.zeros_like(self._obs, device="cpu")
            self._cpu_next_obs_buffer = torch.zeros_like(self._next_obs, device="cpu")
            self._cpu_reward_buffer = torch.zeros_like(self._reward, device="cpu")
            self._cpu_nonterminal_buffer = torch.zeros_like(self._nonterminal, device="cpu")
            self._cpu_gamma_buffer = torch.zeros_like(self._gamma, device="cpu")
            self._cpu_action_buffer = torch.zeros_like(self._action, device="cpu")

    def get_static_batch(self):
        return self._batch

    def empty(self):
        return self.buffer.empty()

    def _timesteps_to_batch(self, timesteps, batch_size):
        # t1 = time.perf_counter()
        _stack_obs_into = self._stack_obs_into
        _compute_n_step = self._compute_n_step

        if self._batch is None:
            obs_shape = timesteps[0].obs.shape
            batch = TensorDict({
                "observation": torch.zeros(batch_size, self.frame_stack, *obs_shape, dtype=torch.float32,
                                           device=self.device),
                "next": TensorDict({
                    "observation": torch.zeros(batch_size, self.frame_stack, *obs_shape, dtype=torch.float32,
                                               device=self.device),
                    "reward": torch.zeros(batch_size, 1, dtype=torch.float32, device=self.device)},
                    batch_size=batch_size, device=self.device),
                "nonterminal": torch.zeros(batch_size, 1, dtype=torch.bool, device=self.device),
                "gamma": torch.ones(batch_size, 1, dtype=torch.float32, device=self.device),
                "action": torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            }, batch_size=batch_size, device=self.device)

            self.set_static_batch(batch)

        # print(f"static batch check time: {time.perf_counter()-t1:f}")
        # t1 = time.perf_counter()

        batch = self._batch

        if self._cpu_obs_buffer is not None:
            obs = self._cpu_obs_buffer
            next_obs = self._cpu_next_obs_buffer
            rewards = self._cpu_reward_buffer
            nonterminals = self._cpu_nonterminal_buffer
            gammas = self._cpu_gamma_buffer
            actions = self._cpu_action_buffer
        else:
            obs = self._obs
            next_obs = self._next_obs
            rewards = self._reward
            nonterminals = self._nonterminal
            gammas = self._gamma
            actions = self._action

        frame_stack = self.frame_stack
        # print(f"local var set time: {time.perf_counter()-t1:f}")
        # loop_start = time.perf_counter()

        # n_step_time = 0
        # frame_copy_time = 0
        # misc_copy_time = 0
        # weakref_call_time = 0
        for i, timestep_list in enumerate(timesteps):
            # This line is necessary because we have to submit a list with only a single element to the PER buffer.
            timestep = timestep_list
            # t1 = time.perf_counter()
            if timestep.needs_n_step:
                _compute_n_step(timestep)
            # n_step_time += time.perf_counter() - t1

            # t1 = time.perf_counter()
            n_step_next = timestep.n_step_next
            if n_step_next is not None:
                n_step_next = n_step_next()
            # weakref_call_time += time.perf_counter() - t1

            # If there is no n-step next timestep.
            # t1 = time.perf_counter()
            if n_step_next is None:

                # If we aren't stacking frames.
                if frame_stack == 1:
                    # Copy the current timestep into the observation buffer.
                    obs[i] = timestep.obs

                # If we do want to stack frames.
                else:
                    # Stack observations backward from the current timestep.
                    _stack_obs_into(timestep, obs[i])

                # Copy the current timestep into the next observation buffer. This is okay because there is no next
                # timestep, so `done` must be True, which means we can't do any bootstrapping.
                next_obs[i] = obs[i]

            # If there is an n-step next timestep.
            else:
                # If we don't want to stack frames.
                if frame_stack == 1:
                    # Copy the current and next timesteps into the observation buffers.
                    obs[i] = timestep.obs
                    next_obs[i] = n_step_next.obs
                else:
                    _stack_obs_into(timestep, obs[i],
                                    next_timestep=n_step_next,
                                    next_buffer=next_obs[i])
            # frame_copy_time += time.perf_counter() - t1

            # t1 = time.perf_counter()
            rewards[i] = timestep.n_step_return
            nonterminals[i] = 1 - timestep.n_step_done
            gammas[i] = timestep.n_step_gamma
            actions[i] = timestep.action
            # misc_copy_time += time.perf_counter() - t1

        # print(f"n_step time: {n_step_time:f}")
        # print(f"weakref call time: {weakref_call_time:f}")
        # print(f"frame copy time: {frame_copy_time:f}")
        # print(f"misc copy time: {misc_copy_time:f}")
        # print(f"loop time: {time.perf_counter()-loop_start:f}")
        if self._cpu_obs_buffer is not None:
            self._obs.copy_(self._cpu_obs_buffer, non_blocking=True)
            self._next_obs.copy_(self._cpu_next_obs_buffer, non_blocking=True)
            self._reward.copy_(self._cpu_reward_buffer, non_blocking=True)
            self._nonterminal.copy_(self._cpu_nonterminal_buffer, non_blocking=True)
            self._gamma.copy_(self._cpu_gamma_buffer, non_blocking=True)
            self._action.copy_(self._cpu_action_buffer, non_blocking=True)

            # print(f"cpu -> gpu copy time: {time.perf_counter()-t1:f}")
        # print()
        return batch

    def _compute_n_step(self, timestep):
        ret = 0
        gamma = 1
        initial_timestep = timestep
        incomplete = False

        for i in range(self.n_step):
            ret += timestep.reward * self.gammas[i]
            gamma = self.gammas[i + 1]
            next_timestep = timestep.next
            incomplete = i != self.n_step - 1

            # First have to check if the weakref is not none and call it.
            if next_timestep is not None and not timestep.truncated and incomplete:
                next_timestep = next_timestep()

                # Have to do the same check after calling the weakref.
                if next_timestep is not None and next_timestep.reward is not None:
                    timestep = next_timestep
                else:
                    break
            else:
                break

        n_step_done = timestep.done
        needs_n_step = incomplete and not n_step_done and not timestep.truncated

        n_step_next_ts = timestep.next
        if type(n_step_next_ts) is Timestep:
            # Truncated timesteps have a true reference to the next timestep, which only contains the final truncated
            # observation. Here we create a weakref to avoid another true reference, so the preceding timestep is
            # the only thing that ever holds a true reference to the truncated timestep (they are never submitted to the buffer).
            n_step_next_ts = weakref.ref(n_step_next_ts)
        else:
            n_step_next_ts = timestep.next

        initial_timestep.n_step_return = ret
        initial_timestep.n_step_gamma = gamma
        initial_timestep.n_step_done = n_step_done
        initial_timestep.n_step_next = n_step_next_ts
        initial_timestep.needs_n_step = needs_n_step

    def _stack_obs_into(self, timestep, observation_buffer, next_timestep=None, next_buffer=None):
        i = self.frame_stack - 1
        while i >= 0 and timestep is not None:
            observation_buffer[i].copy_(timestep.obs, non_blocking=True)

            if next_timestep is not None:
                next_buffer[i].copy_(next_timestep.obs, non_blocking=True)
                if next_timestep.prev is not None:
                    next_timestep = next_timestep.prev()
                else:
                    next_timestep = None

            if timestep.prev is not None:
                timestep = timestep.prev()
            else:
                break

            i -= 1

    def save(self, path):
        buffer_path = os.path.join(path, "experience_buffer")
        os.makedirs(buffer_path, exist_ok=True)

        storage = self.buffer._storage
        timesteps = []
        artificial_truncated_id = -1

        for i in range(len(storage)):
            timestep = storage[i][0]
            serialized = timestep.serialize()

            # If there is a link ahead and that link is a weakref, we may need to truncate this trajectory.
            # However, if the link is a Timestep, we will have already serialized the truncated observation
            # so we can move on.
            if timestep.next is not None and type(timestep.next) is not Timestep:
                next_ts = timestep.next()

                if next_ts is not None:
                    # If this is a partial timestep we must artificially truncate the trajectory this timestep resides
                    # in because when the buffer is loaded we can't recover the state of the data collectors.
                    if next_ts.action is None or next_ts.reward is None:
                        # Mark the next timestep as truncated.
                        next_ts.id = artificial_truncated_id

                        # Remove the weakref.
                        timestep.next = next_ts

                        # Tell the current timestep it is connected to a truncated link.
                        timestep.truncated = True

                        # Re-serialize the timestep so the truncated observation is carried over.
                        serialized = timestep.serialize()

                        # Decrement the artificial truncated id.
                        artificial_truncated_id -= 1

            timesteps += serialized

        with open(os.path.join(buffer_path, "timesteps.pkl"), 'wb') as f:
            pickle.dump(timesteps, f)

        self.buffer._sampler.dumps(buffer_path)
        self.buffer._writer.dumps(buffer_path)

    def load(self, path):
        buffer_path = os.path.join(path, "experience_buffer")
        # load timesteps
        with open(os.path.join(buffer_path, "timesteps.pkl"), 'rb') as f:
            timesteps = pickle.load(f)

        self._load_serialized_timesteps(timesteps)
        self.buffer._sampler.loads(buffer_path)
        self.buffer._writer.loads(buffer_path)

    def _load_serialized_timesteps(self, serialized_timesteps):
        storage = self.buffer._storage
        deserialized_timesteps, _ = Timestep.deserialize_linked_list(serialized_timesteps)
        for i in range(len(deserialized_timesteps)):
            storage[i] = [deserialized_timesteps[i]]


def simple_save_load_test():
    import torch
    from prism.config import LUNAR_LANDER_CFG
    from prism.factory import exp_buffer_factory
    cfg = LUNAR_LANDER_CFG

    def _make_timestep(_ts_id, done, truncated):
        timestep = Timestep(_ts_id)
        timestep.obs = torch.ones(1) * _ts_id
        timestep.action = _ts_id
        timestep.reward = _ts_id
        timestep.done = done
        timestep.truncated = truncated
        return timestep

    exp_buffer = exp_buffer_factory.build_exp_buffer(cfg)
    ts_id = 0

    empty_timestep = Timestep(ts_id)
    ts_id += 1

    populated_timestep = _make_timestep(ts_id, False, False)
    ts_id += 1

    first_linked_timestep = _make_timestep(ts_id, False, False)
    ts_id += 1

    second_linked_timestep = _make_timestep(ts_id, True, False)
    ts_id += 1

    first_linked_timestep.next = weakref.ref(populated_timestep)
    second_linked_timestep.prev = weakref.ref(first_linked_timestep)

    three_link_start = _make_timestep(ts_id, False, False)
    ts_id += 1
    three_link_middle = _make_timestep(ts_id, False, False)
    ts_id += 1
    three_link_end = _make_timestep(ts_id, False, True)
    ts_id += 1
    three_link_truncated = _make_timestep(ts_id, False, False)

    three_link_start.next = weakref.ref(three_link_middle)
    three_link_middle.prev = weakref.ref(three_link_start)
    three_link_middle.next = weakref.ref(three_link_end)
    three_link_end.prev = weakref.ref(three_link_middle)
    three_link_end.next = three_link_truncated

    exp_buffer.extend(empty_timestep)
    exp_buffer.extend(populated_timestep)
    exp_buffer.extend(first_linked_timestep)
    exp_buffer.extend(second_linked_timestep)
    exp_buffer.extend(three_link_start)
    exp_buffer.extend(three_link_middle)
    exp_buffer.extend(three_link_end)

    print("BEFORE SAVE")
    for ts in exp_buffer.buffer._storage:
        print(ts[0])

    exp_buffer.save("test_buffer")

    print("AFTER SAVE")
    for ts in exp_buffer.buffer._storage:
        print(ts[0])

    exp_buffer.empty()
    del exp_buffer

    exp_buffer = exp_buffer_factory.build_exp_buffer(cfg)
    exp_buffer.load("test_buffer")

    print("AFTER LOAD")
    for ts in exp_buffer.buffer._storage:
        print(ts[0])


def complex_save_load_test():
    from torchrl.data.replay_buffers import ReplayBuffer, PrioritizedReplayBuffer
    from torchrl.data import ListStorage
    import torch
    from prism.experience import Timestep
    import weakref

    td_buffer = PrioritizedReplayBuffer(storage=ListStorage(max_size=100), collate_fn=lambda x: x, batch_size=5,
                                        alpha=0.5, beta=0.5)
    buffer = TimestepBuffer(td_buffer, frame_stack=1, device="cpu", n_step=3, gamma=0.99)
    t = Timestep(1)
    t.obs = torch.zeros(3, 4)
    print("filling buffer")
    for i in range(1, 27):  # (1, 27)
        t.reward = i
        t.done = 20 > i > 0 and i % 5 == 0
        t.truncated = i >= 10 and i % 12 == 0
        t.action = torch.ones(1)

        nt = Timestep(i + 1)
        nt.obs = torch.ones(2, 84//2, 84//2) * i

        if t.truncated:
            truncated = Timestep(113 * (i + 1))
            truncated.obs = torch.ones(2, 84//2, 84//2) * -i
            truncated.prev = weakref.ref(t)
            t.next = truncated

        elif not t.done:
            nt.prev = weakref.ref(t)
            t.next = weakref.ref(nt)

        buffer.extend(t)
        t = nt

    print("saving buffer")
    before = buffer.buffer.state_dict()
    buffer.save("test")

    buffer.empty()
    del buffer
    td_buffer = PrioritizedReplayBuffer(storage=ListStorage(max_size=100), collate_fn=lambda x: x, batch_size=5,
                                        alpha=0.5, beta=0.5)
    buffer = TimestepBuffer(td_buffer, frame_stack=1, device="cpu", n_step=3, gamma=0.99)

    print("loading buffer")
    buffer.load("test")
    after = buffer.buffer.state_dict()

    timesteps_before_save = before["_storage"]["_storage"]
    timesteps_after_save = after["_storage"]["_storage"]
    final_timestep = timesteps_after_save[-1][0]
    for i in range(len(timesteps_before_save)):
        bef = timesteps_before_save[i][0]
        aft = timesteps_after_save[i][0]
        assert bef.id == aft.id, """TIMESTEP {} FAILED ID CHECK\n{}\n{}\n""".format(i, bef, aft)
        assert (bef.obs - aft.obs).abs().sum() < 1e-5, """TIMESTEP {} FAILED OBS CHECK\n{}\n{}\n""".format(i, bef, aft)
        assert bef.done == aft.done, """TIMESTEP {} FAILED DONE CHECK\n{}\n{}\n""".format(i, bef, aft)

        # Skip the next of the last timestep because we artificially truncated it.
        if bef.next is not None and bef.id != 26:
            assert bef.truncated == aft.truncated, """TIMESTEP {} FAILED TRUNCATED CHECK\n{}\n{}\n""".format(i, bef,
                                                                                                             aft)
            if bef.truncated:
                assert bef.next == aft.next, """TIMESTEP {} FAILED NEXT CHECK\n{}\n{}\n""".format(i, bef, aft)
            else:
                assert bef.next() == aft.next(), """TIMESTEP {} FAILED NEXT CHECK\n{}\n{}\n""".format(i, bef, aft)
        elif bef.id == 26:
            final_timestep = aft

        # Skip the prev of the first timestep because we didn't save it.
        if bef.prev is not None and bef.id != 17:
            assert aft.prev is not None, """TIMESTEP {} FAILED PREV CHECK\n{}\n{}\n""".format(i, bef, aft)
            assert bef.prev() == aft.prev(), """TIMESTEP {} FAILED PREV CHECK\n{}\n{}\n""".format(i, bef, aft)
        assert bef.needs_n_step == aft.needs_n_step, """TIMESTEP {} FAILED NEEDS_N_STEP CHECK\n{}\n{}\n""".format(i,
                                                                                                                  bef,
                                                                                                                  aft)
    assert type(final_timestep.next) is Timestep, "FINAL TIMESTEP FAILED ARTIFICIAL TRUNCATION CHECK\n{}\n".format(final_timestep)
    print("PASSED SAVE AND LOAD TEST")


if __name__ == "__main__":
    simple_save_load_test()
    complex_save_load_test()


