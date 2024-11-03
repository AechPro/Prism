import torch
import multiprocessing as mp

HALT_PROC_ORDER_FLAG = 1313199
ENV_SHAPES_ORDER_FLAG = 9913131

PROC_WAITING_ACTION_FLAG = 1234567
PROC_SENT_ENV_SHAPES_FLAG = 9876543
PROC_ERROR_FLAG = 17119117


def mp_process(proc_id, shared_memory_pointer, shared_memory_offset):
    import numpy as np
    import time

    mem = np.frombuffer(shared_memory_pointer, dtype=np.float32, offset=shared_memory_offset, count=10_000)
    proc_order_idx = 0
    proc_status_idx = 1
    valid_data_start_idx = 2

    try:
        while mem[proc_order_idx] != HALT_PROC_ORDER_FLAG:
            data = np.random.randn(10_000-2)

            start = valid_data_start_idx
            end = start + data.size
            mem[start:end] = data[:]
            mem[proc_status_idx] = PROC_WAITING_ACTION_FLAG

            while mem[proc_status_idx] == PROC_WAITING_ACTION_FLAG:
                if shared_memory_pointer[proc_order_idx] == HALT_PROC_ORDER_FLAG:
                    return
                time.sleep(0.001)

            received_data = mem[valid_data_start_idx+10:valid_data_start_idx+20]
            print("WORKER RECEIVED ACTION", received_data)

    except:
        mem[proc_status_idx] = PROC_ERROR_FLAG
        import traceback
        print("PROCESS {} ENCOUNTERED AN ERROR".format(proc_id))
        traceback.print_exc()


def test():
    import numpy as np
    import time
    from multiprocessing.sharedctypes import RawArray

    shared_memory = RawArray('f', 10_000)

    can_fork = "forkserver" in mp.get_all_start_methods()
    start_method = "forkserver" if can_fork else "spawn"
    context = mp.get_context(start_method)
    process = context.Process(
        target=mp_process,
        args=(0, shared_memory, 0),
    )
    process.start()

    def _talk_to_process(proc_id, data_to_write, shm_offset_size, flag_idx=None, flag_value=None):
        proc_mem_offset = shm_offset_size*proc_id
        proc_status_idx = proc_mem_offset + 1
        if shared_memory[proc_status_idx] == PROC_ERROR_FLAG:
            return PROC_ERROR_FLAG

        start_idx = proc_mem_offset + 2
        start = start_idx
        end = start + data_to_write.size
        shared_memory[start:end] = data_to_write[:]

        if flag_idx is not None and flag_value is not None:
            shared_memory[flag_idx] = flag_value

        return None

    for i in range(100):
        while shared_memory[1] != PROC_WAITING_ACTION_FLAG:
            time.sleep(0.001)
        obs = shared_memory[2:12]

        print("CENTRAL RECEIVED OBS", obs)
        shared_memory[12:22] = np.ones(10) * i
        shared_memory[1] = 0

    shared_memory[0] = HALT_PROC_ORDER_FLAG
    process.join()


if __name__ == "__main__":
    test()
