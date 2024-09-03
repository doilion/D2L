"""
    这一部分主要说明对应的张量使用运算操作符的一个过程
"""

import torch

# 使用张量间的运算(+、 -、 *、 /)
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

# 用来求对应的幂函数
print(torch.exp(x))

# 将两个张量进行连接，分别沿着行或者列
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1))
print(X == Y)

# 对所有的元素进行一个求和
print(X.sum())
