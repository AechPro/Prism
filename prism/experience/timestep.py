import torch
from dataclasses import dataclass
from typing import TypeVar

Timestep = TypeVar("Timestep")


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
            serialized.append(self.obs.flatten().tolist())
            serialized.append(list(self.obs.shape))
        else:
            serialized.append(None)

        if self.truncated:
            truncated_timestep = self.next
            serialized.append(truncated_timestep.id)
            serialized.append(truncated_timestep.obs.flatten().tolist())
            serialized.append(list(truncated_timestep.obs.shape))
        else:
            serialized.append(None)

        serialized += [self.reward, self.done, self.truncated]

        if self.action is not None:
            serialized.append(self.action)
            # serialized.append(self.action.flatten().tolist())
            # serialized.append(list(self.action.shape))
        else:
            serialized.append(None)

        serialized += [self.n_step_return, self.n_step_gamma, self.n_step_done, self.needs_n_step, self.episodic_reward]
        if self.n_step_next is not None:
            if self.n_step_next() is not None:
                serialized.append(self.n_step_next().id)
            else:
                serialized.append(None)
        else:
            serialized.append(None)

        if self.prev is not None:
            if self.prev() is not None:
                serialized.append(self.prev().id)
            else:
                serialized.append(None)
        else:
            serialized.append(None)

        if self.next is not None:
            if type(self.next) is Timestep:
                next_ts = self.next
            else:
                next_ts = self.next()

            if next_ts is not None:
                serialized.append(next_ts.id)
            else:
                serialized.append(None)
        else:
            serialized.append(None)

        return serialized

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
