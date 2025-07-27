# import torch
# print("CUDA available:", torch.cuda.is_available())
# print("CUDA device count:", torch.cuda.device_count())
# if torch.cuda.is_available():
#     print("Device name:", torch.cuda.get_device_name(0))


import torch
print(torch.__version__)              # версия PyTorch
print(torch.version.cuda)             # версия CUDA, с которой собран PyTorch
print(torch.cuda.is_available())     # доступна ли CUDA
print(torch.cuda.device_count())     # сколько CUDA-устройств видит PyTorch
print(torch.cuda.get_device_name(0)) # название первого GPU