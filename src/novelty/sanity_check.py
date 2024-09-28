import torch
import numpy as np

token1 = [[i for i in range(768)] for j in range(32)]
token1 = np.array(token1)
print(token1.shape)
#token1 = torch.range(1, 32*768)
#token1 = token1.view(32, -1)
token1 = torch.from_numpy(token1)
token1 = token1.view(32, -1)
#token1 = torch.randn((32, 768))
print(token1.size())
print(token1)
outer = torch.einsum("bi,bj->bij", token1, token1)
print(outer.size())
print(outer)