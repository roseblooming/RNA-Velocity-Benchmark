import pynvml
import psutil
import time


def monitor_cpu_memory_usage(pid, result_queue, stop_event, interval=1):
    process = psutil.Process(pid)
    max_memory = 0
    while not stop_event.is_set():
        try:
            memory_info = process.memory_info().rss
            max_memory = max(max_memory, memory_info)
            # print(f"Current Memory: {memory_info / 1024 ** 2:.2f} MB, Max Memory: {max_memory / 1024 ** 2:.2f} MB")
            time.sleep(interval)
        except psutil.NoSuchProcess:
            break
    print(f"Peak Memory Usage for process {pid}: {max_memory / 1024 ** 2:.2f} MB")
    result_queue.put(max_memory / 1024 ** 2)

def monitor_gpu_memory_usage(pid, gpu_index, result_queue, stop_event, interval=1):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    max_memory = 0
    start = False
    while not stop_event.is_set():
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        find = False
        for process in processes:
            if process.pid == pid:
                current_memory = process.usedGpuMemory
                max_memory = max(max_memory, current_memory)
                # print(f"Current GPU Memory Usage: {current_memory / 1024 ** 2:.2f} MB, Max GPU Memory Usage: {max_memory / 1024 ** 2:.2f} MB")
                find = True
                start = True
                break
        if not find and start:
            break
        time.sleep(interval)
    print(f"Peak GPU Memory Usage for process {pid}: {max_memory / 1024 ** 2:.2f} MB")
    pynvml.nvmlShutdown()
    result_queue.put(max_memory / 1024 ** 2)
