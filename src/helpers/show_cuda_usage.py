import torch


def show_cuda_usage():
    print('CUDA Memory Usage')
    print('GPU:       ', torch.cuda.get_device_name(0))
    print('Allocated: ', round(torch.cuda.memory_allocated(device=None)/1024**3, 1), 'GB')
    print('Cached:    ', round(torch.cuda.memory_cached(device=None)/1024**3, 1), 'GB')
    print('Max memory:', round(torch.cuda.max_memory_allocated(device=None)/1024**3, 1), 'GB')
    print('Max Cached:', round(torch.cuda.max_memory_cached(device=None)/1024**3, 1), 'GB')
