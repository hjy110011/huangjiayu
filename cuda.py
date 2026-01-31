import torch

# 1. 查看当前框架已经从显卡那里“圈走”的总空间 (Reserved Memory)
reserved = torch.cuda.memory_reserved(0) / 1024**2
print(f"已申请（预留）显存: {reserved:.2f} MB")

# 2. 查看模型和数据实际占用的空间 (Allocated Memory)
allocated = torch.cuda.memory_allocated(0) / 1024**2
print(f"实际占用显存: {allocated:.2f} MB")

# 3. 查看还能剩下多少预留但未使用的“闲置”空间
free_inside_reserved = (torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)) / 1024**2
print(f"预留中尚未使用的空间: {free_inside_reserved:.2f} MB")