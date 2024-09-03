import torch

X = torch.arange(12, dtype=torch.float32).reshape((3, 4))

# 对张量进行索引和切片
print(X[-1], X[1:3])
print(X[1, 2])

# 为多个元素赋相同的值
X[0:2, :] = 12
print(X)
