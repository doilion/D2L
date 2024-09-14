import math
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from d2l import torch as d2l


def normal(x, mu, sigma):
    """表示的是正态分布"""
    p = 1/math.sqrt(2 * math.pi * sigma**2)
    return p * torch.exp(-1 / (2 * sigma**2) * (x - mu)**2)


# 对于正太分布进行一个可视化的操作
x = torch.arange(-7, 7, 0.01)

# 均值和标准差对
params = [(0, 1), (0, 2), (3, 1)]

d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in params], xlabel='x',
         ylabel='p(x)', figsize=(4.5, 2.5),
         legend=[f'mean {mu}, std {sigma}' for mu, sigma in params])
plt.show()
