import torch

# 广播机制  在大多数情况下,我们将沿着数组中长度为1的轴进行广播
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a, b)
print(a + b)  # 实际最终就是a复制列 b复制对应的行





