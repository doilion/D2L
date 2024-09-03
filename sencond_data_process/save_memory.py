import torch


X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

before = id(Y)
Y = Y + X  # Y变了 指向一个新的地方
print(id(Y) == before)

# 通过使用切片操作来保持就地操作
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# X[:] = X + Y或X += Y 都可以保持一个就地操作
before = id(X)
X += Y
print(id(X) == before)



