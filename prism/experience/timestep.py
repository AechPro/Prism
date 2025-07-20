import time

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
            if type(self.action) is torch.Tensor:
                serialized.append(self.action.item())
            else:
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
        print("deserializing",idx)
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

    @classmethod
    def deserialize_linked_list(cls, serialized_timesteps, timestep_id_map=None):
        """
        Deserialize a list of serialized timesteps into a linked list of Timestep objects.

        Args:
            serialized_timesteps (list): A list of serialized timesteps, where each
                serialized timestep is a list of values as returned by Timestep.serialize.

            timestep_id_map (dict, optional): A map of timestep ids to Timestep objects leftover from a prior call to
                this function. Defaults to None.
        Returns:
            list: A list of linked Timestep objects.
        """

        # Create a list to store the deserialized timesteps, and a map to keep track of the ids.
        # t1 = time.perf_counter()
        timesteps = []
        incomplete_timestep_id_map = {}
        if timestep_id_map is None:
            timestep_id_map = {}

        # Iterate over the serialized timesteps, and for each one, deserialize it and store it and its links in the map.
        idx = 0
        total_timesteps = 0
        while idx < len(serialized_timesteps):
            timestep, links, idx = Timestep.deserialize(serialized_timesteps, idx)
            timestep_id_map[timestep.id] = (timestep, links)
            total_timesteps += 1
            print("Deserialized timestep", timestep.id, total_timesteps)

        # Iterate over the map, and for each timestep, set its prev, n_step_next, and next links based on the links stored in the map.
        idx = 0
        # known_ids = list(timestep_id_map.keys())
        for ts_id, data in timestep_id_map.items():
            timestep, links = data

            # If there are only two links the timestep has either been truncated or does not have a next link.
            if len(links) == 2:
                n_step_next_id, prev_id = links
                next_id = None
            else:
                n_step_next_id, prev_id, next_id = links

            waiting_links = [None, None, None]
            no_links_remain = True

            # Set the prev link.
            if prev_id is not None:
                prev_timestep = timestep_id_map.get(prev_id, None)
                if prev_timestep is not None:
                    timestep.prev = weakref.ref(prev_timestep[0])
                else:
                    waiting_links[1] = prev_id
                    no_links_remain = False

            # Set the n_step_next link.
            if n_step_next_id is not None:
                n_step_next_timestep = timestep_id_map.get(n_step_next_id, None)
                if n_step_next_timestep is not None:
                    timestep.n_step_next = weakref.ref(n_step_next_timestep[0])
                else:
                    waiting_links[0] = n_step_next_id
                    no_links_remain = False

            # Set the next link if necessary.
            if next_id is not None and not timestep.truncated:
                next_timestep = timestep_id_map.get(next_id, None)
                if next_timestep is not None:
                    timestep.next = weakref.ref(next_timestep[0])
                else:
                    waiting_links[2] = next_id
                    no_links_remain = False

            if no_links_remain:
                timesteps.append(timestep)
            else:
                incomplete_timestep_id_map[ts_id] = (timestep, waiting_links)

            idx += 1

        timestep_id_map.clear()
        # print("Deserialized", len(timesteps), "timesteps in", time.perf_counter() - t1, "seconds")
        return timesteps, incomplete_timestep_id_map

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
