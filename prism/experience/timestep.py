import torch
from dataclasses import dataclass
from typing import TypeVar
import weakref

Timestep = TypeVar("Timestep")
NULL_VALUE = -1313


@dataclass()
class Timestep(object):
    id: int
    obs: torch.Tensor = None
    reward: float = None
    done: bool = None
    truncated: bool = None
    action: int = None
    n_step_return: float = None
    n_step_gamma: float = None
    n_step_done: bool = None
    needs_n_step: bool = True
    episodic_reward: float = 0

    n_step_next: Timestep = None
    prev: Timestep = None
    next: Timestep = None

    def serialize(self):
        serialized = [self.id]

        if self.obs is not None:
            n_vals_in_obs = self.obs.numel()
            serialized.append(n_vals_in_obs)
            serialized += self.obs.flatten().tolist()

            n_vals_in_shape = len(self.obs.shape)
            serialized.append(n_vals_in_shape)
            serialized += list(self.obs.shape)
        else:
            serialized.append(NULL_VALUE)

        if self.truncated:
            truncated_timestep = self.next
            serialized.append(truncated_timestep.id)

            n_vals_in_obs = truncated_timestep.obs.numel()
            serialized.append(n_vals_in_obs)
            serialized += truncated_timestep.obs.flatten().tolist()

            n_vals_in_shape = len(truncated_timestep.obs.shape)
            serialized.append(n_vals_in_shape)
            serialized += list(truncated_timestep.obs.shape)
        else:
            serialized.append(NULL_VALUE)

        serialized += [self.reward, self.done, self.truncated]

        if self.action is not None:
            serialized.append(self.action)
        else:
            serialized.append(NULL_VALUE)

        serialized += [self.n_step_return, self.n_step_gamma, self.n_step_done, self.needs_n_step, self.episodic_reward]
        if self.n_step_next is not None:
            if self.n_step_next() is not None:
                serialized.append(self.n_step_next().id)
            else:
                serialized.append(NULL_VALUE)
        else:
            serialized.append(NULL_VALUE)

        if self.prev is not None:
            if self.prev() is not None:
                serialized.append(self.prev().id)
            else:
                serialized.append(NULL_VALUE)
        else:
            serialized.append(NULL_VALUE)

        if self.next is not None:
            if type(self.next) is Timestep:
                next_ts = self.next
            else:
                next_ts = self.next()

            if next_ts is not None:
                serialized.append(next_ts.id)
            else:
                serialized.append(NULL_VALUE)
        else:
            serialized.append(NULL_VALUE)

        for i in range(len(serialized)):
            if serialized[i] is None:
                serialized[i] = NULL_VALUE

        return serialized

    @classmethod
    def deserialize(cls, serialized_timestep, idx):
        timestep_id = int(serialized_timestep[idx])
        idx += 1

        if serialized_timestep[idx] != NULL_VALUE:
            n_vals_in_obs = int(serialized_timestep[idx])
            idx += 1

            obs_list = serialized_timestep[idx:idx + n_vals_in_obs]
            idx += n_vals_in_obs

            n_vals_in_shape = int(serialized_timestep[idx])
            idx += 1

            shape_list = serialized_timestep[idx:idx + n_vals_in_shape]
            idx += n_vals_in_shape

            obs = torch.as_tensor(obs_list,
                                  dtype=torch.float32,
                                  device="cpu").reshape([int(arg) for arg in shape_list])
        else:
            obs = None
            idx += 1

        if serialized_timestep[idx] != NULL_VALUE:
            truncated_id = int(serialized_timestep[idx])
            idx += 1

            n_vals_in_obs = int(serialized_timestep[idx])
            idx += 1

            obs_list = serialized_timestep[idx:idx + n_vals_in_obs]
            idx += n_vals_in_obs

            n_vals_in_shape = int(serialized_timestep[idx])
            idx += 1

            shape_list = serialized_timestep[idx:idx + n_vals_in_shape]
            idx += n_vals_in_shape

            truncated_obs = torch.as_tensor(obs_list,
                                            dtype=torch.float32,
                                            device="cpu").reshape([int(arg) for arg in shape_list])

            truncated_timestep = Timestep(id=truncated_id, obs=truncated_obs)
        else:
            truncated_timestep = None
            idx += 1

        reward = float(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        done = bool(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        truncated = bool(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1

        action = int(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1

        n_step_return = float(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        n_step_gamma = float(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        n_step_done = bool(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        needs_n_step = bool(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        episodic_reward = float(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        n_step_next_id = int(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        prev_id = int(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1
        next_id = int(serialized_timestep[idx]) if serialized_timestep[idx] != NULL_VALUE else None
        idx += 1

        timestep = cls(id=timestep_id, obs=obs, reward=reward, done=done, truncated=truncated, action=action,
                       n_step_return=n_step_return, n_step_gamma=n_step_gamma, n_step_done=n_step_done,
                       needs_n_step=needs_n_step, episodic_reward=episodic_reward, n_step_next=None, prev=None,
                       next=truncated_timestep)

        if truncated_timestep is None:
            required_links = (n_step_next_id, prev_id, next_id)
        else:
            required_links = (n_step_next_id, prev_id)
            truncated_timestep.prev = weakref.ref(timestep)

        return timestep, required_links, idx

    def __repr__(self):
        return (f"Timestep({self.id}, "
                f"{self.episodic_reward}, "
                f"{self.obs}, "
                f"{self.reward}, "
                f"{self.done}, "
                f"{self.truncated}, "
                f"{self.action}, "
                f"{self.n_step_return}, "
                f"{self.n_step_gamma}, "
                f"{self.n_step_done}, "
                f"{self.needs_n_step}, "
                f"{self.n_step_next is not None}, "
                f"{self.prev is not None}, "
                f"{self.next is not None})")

    def __eq__(self, other):
        return self.id == other.id
