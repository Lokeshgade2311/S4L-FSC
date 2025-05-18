import numpy as np
import torch

# numpy 噪声
np_noise = np.random.normal(loc=0, scale=1, size=(1000,))
print("numpy: mean={}, std={}".format(np.mean(np_noise), np.std(np_noise)))

# PyTorch 噪声（CPU）
pt_noise = torch.randn(1000)
print("PyTorch: mean={}, std={}".format(pt_noise.mean().item(), pt_noise.std().item()))