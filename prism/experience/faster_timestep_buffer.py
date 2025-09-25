import torch
import weakref
from collections import deque
from tensordict import TensorDict
from prism.experience import Timestep

class FasterTimestepBuffer(object):
    """
    High-throughput replay wrapper around a TorchRL buffer (ReplayBuffer or PER)
    that keeps a shared frame store (ring of raw observations) and caches
    per-timestep stacked frame-ids to enable fast batch sampling via vectorized
    gathers. Saving/loading is intentionally omitted in this prototype.

    Notes
    - Single writer assumed (the learner thread calling extend()).
    - We maintain our own write cursor to mirror TorchRL storage cycling so we
      can decrement frame refcounts when a slot is overwritten.
    - We set timestep.obs=None after ingest to reduce memory pressure; we add a
      dynamic attribute "frame_id" to the Timestep instance.
    """

    def __init__(self, torchrl_buffer, frame_stack=1, device="cpu", n_step=3, gamma=0.99):
        self.buffer = torchrl_buffer
        self.device = device
        self.frame_stack = int(frame_stack)
        self.n_step = int(n_step)
        self.gammas = [gamma ** i for i in range(self.n_step + 1)]

        # Underlying storage (ListStorage in this codebase) and capacity
        self._storage = self.buffer._storage
        self._capacity = self._resolve_capacity()
        self._write_idx = 0
        self._filled = 0

        # Frame store (lazy init on first extend)
        self._obs_shape = None
        self._frames_capacity = None
        self._frame_store = None   # [frames_capacity+1, *obs_shape], index 0 is zero-sentinel
        self._frame_refcount = None  # torch.int64
        self._next_frame_id = 1
        self._free_frame_ids = deque()

        # Caches and bookkeeping
        self._slot_bundles = [None for _ in range(self._capacity)]  # per storage slot
        self._tsid_to_slot = {}
        self._obs_stack_cache = {}  # ts.id -> list[int] of length frame_stack
        # Per-chain pending queues for submit-time N-step finalization
        self._pending_chains = {}  # chain_id -> deque[Timestep]
        self._ts_to_chain = {}     # ts.id -> chain_id

        # Slot-level tensors for vectorized sampling
        self._slot_obs_ids = torch.zeros(self._capacity, self.frame_stack, dtype=torch.long, device='cpu')
        self._slot_next_ids = torch.zeros(self._capacity, self.frame_stack, dtype=torch.long, device='cpu')
        self._slot_reward = torch.zeros(self._capacity, 1, dtype=torch.float32, device='cpu')
        self._slot_gamma = torch.ones(self._capacity, 1, dtype=torch.float32, device='cpu')
        self._slot_nonterminal = torch.zeros(self._capacity, 1, dtype=torch.bool, device='cpu')
        self._slot_action = torch.zeros(self._capacity, 1, dtype=torch.long, device='cpu')
        self._slot_finalized = torch.zeros(self._capacity, dtype=torch.bool, device='cpu')

        # Static batch buffers
        self._batch = None
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
        # Reusable CPU id batches for frame gathers
        self._cpu_obs_ids_batch = None
        self._cpu_next_ids_batch = None

    # ----------------------------- Public API -----------------------------
    def extend(self, timestep: Timestep):
        # Lazy init of frame store
        if self._obs_shape is None:
            self._obs_shape = tuple(timestep.obs.shape)
            self._init_frame_store()

        # If overwriting an existing slot, release its frame refs
        old = self._slot_bundles[self._write_idx]
        if old is not None:
            self._release_slot_bundle(old)
            self._tsid_to_slot.pop(old['ts_id'], None)
        # Mark the slot as unfinalized on overwrite
        self._slot_finalized[self._write_idx] = False

        # Assign frame_id for this timestep and copy obs into frame store
        frame_id = self._ensure_frame_id_for_ts(timestep)
        # Cache backward obs-stack frame ids (no ref inc here; inc only for slot ownership)
        obs_ids = self._build_stack_ids(timestep)
        self._obs_stack_cache[timestep.id] = obs_ids

        # Create and register slot bundle (owns obs_ids; next_ids will be filled when finalized)
        bundle = {
            'ts_id': timestep.id,
            'obs_ids': obs_ids,
            'next_ids': None,
            'reward': None,
            'gamma': None,
            'nonterminal': None,
            'finalized': False,
        }
        self._slot_bundles[self._write_idx] = bundle
        self._tsid_to_slot[timestep.id] = self._write_idx
        self._inc_frame_refs(obs_ids)

        # Update slot-level obs ids and action
        row = self._slot_obs_ids[self._write_idx]
        for k in range(self.frame_stack):
            row[k] = obs_ids[k]
        self._slot_action[self._write_idx, 0] = int(timestep.action) if timestep.action is not None else 0

        # Update per-chain pending deques and finalize as much as possible
        self._update_pending_chains_and_finalize(timestep)

        # Push to underlying TorchRL buffer (stores the Timestep object)
        self.buffer.extend([timestep])

        # Advance cursor
        self._write_idx = (self._write_idx + 1) % self._capacity
        if self._filled < self._capacity:
            self._filled += 1

    @torch.no_grad()
    def sample(self, batch_size=None, return_info=False):
        data = self.buffer.sample(batch_size=batch_size, return_info=True)
        timesteps, info = data
        if batch_size is None:
            batch_size = self.buffer._batch_size

        # Ensure static batch exists
        if self._batch is None:
            self._init_static_batch(batch_size)

        # CPU staging views
        if self._cpu_obs_buffer is not None:
            obs = self._cpu_obs_buffer
            next_obs = self._cpu_next_obs_buffer
            rewards = self._cpu_reward_buffer
            nonterm = self._cpu_nonterminal_buffer
            gammas = self._cpu_gamma_buffer
            actions = self._cpu_action_buffer
        else:
            obs = self._obs
            next_obs = self._next_obs
            rewards = self._reward
            nonterm = self._nonterminal
            gammas = self._gamma
            actions = self._action

        # Build frame-id batches and scalars
        indices_in = info['index']
        if isinstance(indices_in, torch.Tensor):
            idx_t = indices_in.to(dtype=torch.long, device='cpu', copy=False)
        else:
            idx_t = torch.as_tensor(indices_in, dtype=torch.long, device='cpu')

        # Reuse pre-allocated id buffers to avoid alloc overhead
        obs_ids_batch = self._cpu_obs_ids_batch
        next_ids_batch = self._cpu_next_ids_batch

        # Vectorized fast path for finalized samples
        finalized_mask = self._slot_finalized.index_select(0, idx_t)
        if finalized_mask.any():
            idx_final = idx_t[finalized_mask]
            obs_ids_batch[finalized_mask] = self._slot_obs_ids.index_select(0, idx_final)
            next_ids_batch[finalized_mask] = self._slot_next_ids.index_select(0, idx_final)
            rewards[finalized_mask] = self._slot_reward.index_select(0, idx_final)
            gammas[finalized_mask] = self._slot_gamma.index_select(0, idx_final)
            nonterm[finalized_mask] = self._slot_nonterminal.index_select(0, idx_final)
            actions[finalized_mask] = self._slot_action.index_select(0, idx_final)

        # Slow path for unfinalized samples (rare edge cases)
        if (~finalized_mask).any():
            idx_slow_positions = (~finalized_mask).nonzero(as_tuple=False).squeeze(1)
            for pos in idx_slow_positions.tolist():
                t = timesteps[pos][0]
                # Finalize this timestep now (also writes slot tensors)
                self._finalize_single_timestep(t)
                slot_idx = self._tsid_to_slot.get(t.id)
                # Fill row from slot tensors
                obs_ids_batch[pos] = self._slot_obs_ids[slot_idx]
                next_ids_batch[pos] = self._slot_next_ids[slot_idx]
                rewards[pos] = self._slot_reward[slot_idx]
                gammas[pos] = self._slot_gamma[slot_idx]
                nonterm[pos] = self._slot_nonterminal[slot_idx]
                actions[pos] = self._slot_action[slot_idx]

        # Vectorized gather from frame store (CPU) using index_select for lower overhead
        flat_obs_ids = obs_ids_batch.view(-1)
        flat_next_ids = next_ids_batch.view(-1)
        obs_cpu_flat = self._frame_store.index_select(0, flat_obs_ids)
        next_cpu_flat = self._frame_store.index_select(0, flat_next_ids)
        obs_cpu = obs_cpu_flat.view(batch_size, self.frame_stack, *self._obs_shape)
        next_cpu = next_cpu_flat.view(batch_size, self.frame_stack, *self._obs_shape)
        obs.copy_(obs_cpu, non_blocking=True)
        next_obs.copy_(next_cpu, non_blocking=True)

        # H2D if needed
        if self._cpu_obs_buffer is not None:
            self._obs.copy_(self._cpu_obs_buffer, non_blocking=True)
            self._next_obs.copy_(self._cpu_next_obs_buffer, non_blocking=True)
            self._reward.copy_(self._cpu_reward_buffer, non_blocking=True)
            self._nonterminal.copy_(self._cpu_nonterminal_buffer, non_blocking=True)
            self._gamma.copy_(self._cpu_gamma_buffer, non_blocking=True)
            self._action.copy_(self._cpu_action_buffer, non_blocking=True)

        if return_info:
            return self._batch, info
        return self._batch

    def update_priority(self, indices, priorities):
        if hasattr(self.buffer, 'update_priority'):
            self.buffer.update_priority(indices, priorities)

    def set_static_batch(self, batch):
        self._batch = batch
        self._obs = self._batch['observation']
        self._next_obs = self._batch['next']['observation']
        self._reward = self._batch['next']['reward']
        self._nonterminal = self._batch['nonterminal']
        self._gamma = self._batch['gamma']
        self._action = self._batch['action']

        if 'cpu' not in self.device:
            # Allocate pinned CPU staging
            self._cpu_obs_buffer = torch.empty_like(self._obs, device='cpu', pin_memory=True)
            self._cpu_next_obs_buffer = torch.empty_like(self._next_obs, device='cpu', pin_memory=True)
            self._cpu_reward_buffer = torch.empty_like(self._reward, device='cpu', pin_memory=True)
            self._cpu_nonterminal_buffer = torch.empty_like(self._nonterminal, device='cpu', pin_memory=True)
            self._cpu_gamma_buffer = torch.empty_like(self._gamma, device='cpu', pin_memory=True)
            self._cpu_action_buffer = torch.empty_like(self._action, device='cpu', pin_memory=True)
        # Allocate reusable CPU id batches (pin for faster H2D if needed)
        pin = 'cpu' not in self.device
        bs = self._batch.batch_size[0] if hasattr(self._batch, 'batch_size') else self._obs.shape[0]
        self._cpu_obs_ids_batch = torch.empty((bs, self.frame_stack), dtype=torch.long, device='cpu', pin_memory=pin)
        self._cpu_next_ids_batch = torch.empty((bs, self.frame_stack), dtype=torch.long, device='cpu', pin_memory=pin)

    def get_static_batch(self):
        return self._batch

    def empty(self):
        # Release all frame refs
        for bundle in self._slot_bundles:
            if bundle is not None:
                self._release_slot_bundle(bundle)
        self._slot_bundles = [None for _ in range(self._capacity)]
        self._tsid_to_slot.clear()
        self._obs_stack_cache.clear()
        self.buffer.empty()

    # ----------------------------- Internals -----------------------------
    def _init_frame_store(self):
        # Frames capacity: buffer capacity plus a fixed safety headroom.
        # Keep memory bounded and cyclic like TorchRL; add margin for frame stacking and n-step.
        headroom = max(1024, (self.frame_stack + self.n_step) * 2 + max(256, self._capacity // 16))
        self._frames_capacity = int(self._capacity + headroom)
        pin = 'cpu' not in self.device
        self._frame_store = torch.zeros((self._frames_capacity + 1, *self._obs_shape),
                                        dtype=torch.float32, device='cpu', pin_memory=pin)
        self._frame_refcount = torch.zeros((self._frames_capacity + 1,), dtype=torch.int64, device='cpu')

    def _alloc_frame_id(self):
        if self._free_frame_ids:
            return self._free_frame_ids.popleft()
        if self._next_frame_id <= self._frames_capacity:
            fid = self._next_frame_id
            self._next_frame_id += 1
            return fid
        # Fallback scan (rare)
        ref = self._frame_refcount
        zeros = (ref == 0).nonzero(as_tuple=False)
        for z in zeros:
            fid = int(z.item())
            if fid != 0:  # skip sentinel
                return fid
        # No free slots found; fixed-capacity policy — keep memory bounded
        raise RuntimeError("Frame store exhausted: increase replay capacity or headroom, or reduce frame_stack/n_step.")

    def _inc_frame_refs(self, ids):
        # ids: list[int]
        if not ids:
            return
        ids_t = torch.as_tensor(ids, dtype=torch.long, device='cpu')
        mask = ids_t != 0  # skip sentinel zero
        if mask.any():
            self._frame_refcount.index_add_(0, ids_t[mask], torch.ones_like(ids_t[mask], dtype=self._frame_refcount.dtype))

    def _dec_frame_refs(self, ids):
        if not ids:
            return
        ids_t = torch.as_tensor(ids, dtype=torch.long, device='cpu')
        mask = ids_t != 0
        if mask.any():
            self._frame_refcount.index_add_(0, ids_t[mask], -torch.ones_like(ids_t[mask], dtype=self._frame_refcount.dtype))
            # Collect zeros and add to free list
            touched = ids_t[mask].unique()
            for fid in touched.tolist():
                if fid != 0 and self._frame_refcount[fid] <= 0:
                    self._free_frame_ids.append(int(fid))

    def _release_slot_bundle(self, bundle):
        self._dec_frame_refs(bundle['obs_ids'])
        if bundle.get('next_ids') is not None:
            self._dec_frame_refs(bundle['next_ids'])

    def _ensure_frame_id_for_ts(self, ts: Timestep):
        fid = getattr(ts, 'frame_id', None)
        if fid is not None:
            return fid
        # If obs is available (e.g., truncated timestep), allocate and copy
        if ts.obs is None:
            # If we don't have obs, we expect this ts to already have been ingested earlier
            raise RuntimeError("Timestep without frame_id or obs encountered.")
        fid = self._alloc_frame_id()
        src = ts.obs.detach().to(dtype=torch.float32, device='cpu', copy=False)
        self._frame_store[fid].copy_(src, non_blocking=False)
        setattr(ts, 'frame_id', fid)
        # Free original obs to save memory
        ts.obs = None
        return fid

    def _build_stack_ids(self, ts: Timestep):
        ids = [0] * self.frame_stack
        i = self.frame_stack - 1
        cur = ts
        while i >= 0 and cur is not None:
            fid = getattr(cur, 'frame_id', None)
            if fid is None:
                # For truncated timesteps, allocate on-demand
                if cur.obs is not None:
                    fid = self._ensure_frame_id_for_ts(cur)
                else:
                    # Should not happen for previously ingested steps
                    fid = 0
            ids[i] = int(fid)
            # Move prev
            if cur.prev is not None:
                cur = cur.prev()
            else:
                break
            i -= 1
        return ids

    def _compute_n_step(self, timestep: Timestep):
        ret = 0.0
        gamma = 1.0
        initial = timestep
        incomplete = False

        cur = timestep
        for i in range(self.n_step):
            if cur.reward is None:
                break
            ret += float(cur.reward) * self.gammas[i]
            gamma = self.gammas[i + 1]
            next_ts = cur.next
            incomplete = i != self.n_step - 1
            if next_ts is not None and not cur.truncated and incomplete:
                if not isinstance(next_ts, Timestep):
                    next_ts = next_ts()
                if next_ts is not None and next_ts.reward is not None:
                    cur = next_ts
                else:
                    break
            else:
                break

        n_step_done = bool(cur.done)
        needs_n_step = incomplete and (not n_step_done) and (not cur.truncated)

        n_step_next = cur.next
        if isinstance(n_step_next, Timestep):
            n_step_next = weakref.ref(n_step_next)

        initial.n_step_return = ret
        initial.n_step_gamma = gamma
        initial.n_step_done = n_step_done
        initial.n_step_next = n_step_next
        initial.needs_n_step = needs_n_step

    # ----------------------------- Submit-time N-step -----------------------------
    def _update_pending_chains_and_finalize(self, timestep: Timestep):
        # Determine chain id using prev link if available
        prev_ts = timestep.prev() if timestep.prev is not None else None
        if prev_ts is not None and prev_ts.id in self._ts_to_chain:
            chain_id = self._ts_to_chain[prev_ts.id]
        else:
            chain_id = timestep.id  # start new chain
            self._pending_chains.setdefault(chain_id, deque())
        dq = self._pending_chains[chain_id]
        dq.append(timestep)
        self._ts_to_chain[timestep.id] = chain_id

        # Finalize as much as possible with available lookahead
        self._try_finalize_head(dq)

        # On boundary, flush remaining
        if timestep.done or timestep.truncated:
            self._flush_chain(chain_id)

    def _try_finalize_head(self, dq: deque):
        while len(dq) > 0:
            head = dq[0]
            ret = 0.0
            gamma_val = 1.0
            steps_used = 0
            done_flag = False
            trunc_flag = False
            last_idx = 0
            for k in range(self.n_step):
                if k >= len(dq):
                    break
                cur = dq[k]
                if cur.reward is None:
                    break
                ret += float(cur.reward) * self.gammas[k]
                gamma_val = self.gammas[k + 1]
                steps_used = k + 1
                last_idx = k
                if cur.done:
                    done_flag = True
                    break
                if cur.truncated:
                    trunc_flag = True
                    break

            # Decide if we can finalize head now
            can_finalize = False
            next_ids = None
            if done_flag:
                can_finalize = True
                next_ids = self._obs_stack_cache.get(head.id)
                if next_ids is None:
                    next_ids = self._build_stack_ids(head)
                    self._obs_stack_cache[head.id] = next_ids
            elif trunc_flag:
                can_finalize = True
                trunc_ts = dq[last_idx].next  # real Timestep for truncated next
                if not isinstance(trunc_ts, Timestep):
                    # Should be a real Timestep by design
                    trunc_ts = trunc_ts()
                if trunc_ts is not None:
                    self._ensure_frame_id_for_ts(trunc_ts)
                    next_ids = self._obs_stack_cache.get(trunc_ts.id)
                    if next_ids is None:
                        next_ids = self._build_stack_ids(trunc_ts)
                        self._obs_stack_cache[trunc_ts.id] = next_ids
                else:
                    next_ids = self._obs_stack_cache.get(head.id) or self._build_stack_ids(head)
            elif steps_used == self.n_step and len(dq) >= self.n_step + 1:
                can_finalize = True
                nxt = dq[self.n_step]
                self._ensure_frame_id_for_ts(nxt)
                next_ids = self._obs_stack_cache.get(nxt.id)
                if next_ids is None:
                    next_ids = self._build_stack_ids(nxt)
                    self._obs_stack_cache[nxt.id] = next_ids

            if not can_finalize:
                break

            # Fill Timestep head fields
            cur = dq[last_idx]
            head.n_step_return = ret
            head.n_step_gamma = gamma_val
            head.n_step_done = bool(cur.done)
            if cur.done:
                head.n_step_next = None
            elif cur.truncated:
                head.n_step_next = weakref.ref(cur.next) if isinstance(cur.next, Timestep) else cur.next
            else:
                head.n_step_next = weakref.ref(dq[self.n_step])
            head.needs_n_step = False

            # Update slot bundle
            slot_idx = self._tsid_to_slot.get(head.id)
            if slot_idx is not None:
                bundle = self._slot_bundles[slot_idx]
                if bundle is not None and not bundle.get('finalized'):
                    bundle['next_ids'] = next_ids
                    bundle['reward'] = float(head.n_step_return)
                    bundle['gamma'] = float(head.n_step_gamma)
                    bundle['nonterminal'] = int(not bool(head.n_step_done))
                    bundle['finalized'] = True
                    self._inc_frame_refs(next_ids)
                    # Write slot tensors and mark finalized
                    row_next = self._slot_next_ids[slot_idx]
                    for j in range(self.frame_stack):
                        row_next[j] = next_ids[j]
                    self._slot_reward[slot_idx, 0] = float(head.n_step_return)
                    self._slot_gamma[slot_idx, 0] = float(head.n_step_gamma)
                    self._slot_nonterminal[slot_idx, 0] = int(not bool(head.n_step_done))
                    self._slot_finalized[slot_idx] = True

            # Pop head and continue
            dq.popleft()

    def _flush_chain(self, chain_id):
        dq = self._pending_chains.get(chain_id)
        if dq is None:
            return
        # Finalize remaining heads even if fewer than n steps remain
        while len(dq) > 0:
            head = dq[0]
            ret = 0.0
            gamma_val = 1.0
            last_cur = head
            for k in range(min(self.n_step, len(dq))):
                cur = dq[k]
                if cur.reward is None:
                    break
                ret += float(cur.reward) * self.gammas[k]
                gamma_val = self.gammas[k + 1]
                last_cur = cur
                if cur.done or cur.truncated:
                    break

            # Determine next_ids
            if last_cur.done:
                next_ids = self._obs_stack_cache.get(head.id) or self._build_stack_ids(head)
                head.n_step_next = None
                head.n_step_done = True
            elif last_cur.truncated:
                trunc_ts = last_cur.next
                if not isinstance(trunc_ts, Timestep):
                    trunc_ts = trunc_ts()
                self._ensure_frame_id_for_ts(trunc_ts)
                next_ids = self._obs_stack_cache.get(trunc_ts.id) or self._build_stack_ids(trunc_ts)
                self._obs_stack_cache[trunc_ts.id] = next_ids
                head.n_step_next = weakref.ref(trunc_ts)
                head.n_step_done = False
            else:
                # Fewer than n steps but boundary reached; treat as terminal for bootstrap mask
                next_ids = self._obs_stack_cache.get(head.id) or self._build_stack_ids(head)
                head.n_step_next = None
                head.n_step_done = True

            head.n_step_return = ret
            head.n_step_gamma = gamma_val
            head.needs_n_step = False

            slot_idx = self._tsid_to_slot.get(head.id)
            if slot_idx is not None:
                bundle = self._slot_bundles[slot_idx]
                if bundle is not None and not bundle.get('finalized'):
                    bundle['next_ids'] = next_ids
                    bundle['reward'] = float(head.n_step_return)
                    bundle['gamma'] = float(head.n_step_gamma)
                    bundle['nonterminal'] = int(not bool(head.n_step_done))
                    bundle['finalized'] = True
                    self._inc_frame_refs(next_ids)
                    # Write slot tensors and mark finalized
                    row_next = self._slot_next_ids[slot_idx]
                    for j in range(self.frame_stack):
                        row_next[j] = next_ids[j]
                    self._slot_reward[slot_idx, 0] = float(head.n_step_return)
                    self._slot_gamma[slot_idx, 0] = float(head.n_step_gamma)
                    self._slot_nonterminal[slot_idx, 0] = int(not bool(head.n_step_done))
                    self._slot_finalized[slot_idx] = True

            dq.popleft()
        # Remove chain entry
        self._pending_chains.pop(chain_id, None)

    def _finalize_single_timestep(self, timestep: Timestep):
        """Finalize a single timestep and populate slot tensors if not already finalized."""
        if getattr(timestep, 'needs_n_step', True):
            self._compute_n_step(timestep)
        slot_idx = self._tsid_to_slot.get(timestep.id)
        if slot_idx is None:
            return
        bundle = self._slot_bundles[slot_idx]
        if bundle is None or bundle.get('finalized'):
            return
        # Determine next_ids
        next_ts = timestep.n_step_next
        if next_ts is None:
            next_ids = self._obs_stack_cache.get(timestep.id)
            if next_ids is None:
                next_ids = self._build_stack_ids(timestep)
                self._obs_stack_cache[timestep.id] = next_ids
        else:
            if not isinstance(next_ts, Timestep):
                next_ts = next_ts()
            if next_ts is None:
                next_ids = self._obs_stack_cache.get(timestep.id) or self._build_stack_ids(timestep)
            else:
                self._ensure_frame_id_for_ts(next_ts)
                next_ids = self._obs_stack_cache.get(next_ts.id)
                if next_ids is None:
                    next_ids = self._build_stack_ids(next_ts)
                    self._obs_stack_cache[next_ts.id] = next_ids
        # Update bundle and slot tensors
        bundle['next_ids'] = next_ids
        bundle['reward'] = float(timestep.n_step_return)
        bundle['gamma'] = float(timestep.n_step_gamma)
        bundle['nonterminal'] = int(not bool(timestep.n_step_done))
        bundle['finalized'] = True
        self._inc_frame_refs(next_ids)
        row_next = self._slot_next_ids[slot_idx]
        for j in range(self.frame_stack):
            row_next[j] = next_ids[j]
        self._slot_reward[slot_idx, 0] = float(timestep.n_step_return)
        self._slot_gamma[slot_idx, 0] = float(timestep.n_step_gamma)
        self._slot_nonterminal[slot_idx, 0] = int(not bool(timestep.n_step_done))
        self._slot_finalized[slot_idx] = True

    # ----------------------------- Utilities -----------------------------
    def _resolve_capacity(self):
        # Try common attribute names from torchrl ListStorage/Writer
        for obj in (self._storage, getattr(self.buffer, "_writer", None)):
            if obj is None:
                continue
            for name in ("max_size", "_max_size", "capacity", "_capacity"):
                cap = getattr(obj, name, None)
                if cap is not None:
                    try:
                        return int(cap)
                    except Exception:
                        pass
        # Fallback: try buffer.state_dict() metadata
        try:
            sd = self.buffer.state_dict()
            cap = sd.get('_writer', {}).get('max_size', None)
            if cap is not None:
                return int(cap)
        except Exception:
            pass
        raise RuntimeError("Unable to resolve replay capacity from TorchRL buffer.")

    def _init_static_batch(self, batch_size):
        bs = batch_size
        obs = torch.zeros(bs, self.frame_stack, *self._obs_shape, dtype=torch.float32, device=self.device)
        next_obs = torch.zeros_like(obs)
        rew = torch.zeros(bs, 1, dtype=torch.float32, device=self.device)
        nonterm = torch.zeros(bs, 1, dtype=torch.bool, device=self.device)
        gamma = torch.ones(bs, 1, dtype=torch.float32, device=self.device)
        act = torch.zeros(bs, 1, dtype=torch.long, device=self.device)

        batch = TensorDict({
            'observation': obs,
            'next': TensorDict({'observation': next_obs, 'reward': rew}, batch_size=bs, device=self.device),
            'nonterminal': nonterm,
            'gamma': gamma,
            'action': act,
        }, batch_size=bs, device=self.device)

        self.set_static_batch(batch)