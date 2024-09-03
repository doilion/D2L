import torch

# 出来的是0 到 12 前的一个张量
x = torch.arange(12)
print(x)
print(type(x))  # torch.Tensor()

# 通过张量的属性来访问张量的形状
print(x.shape)

# 查看张量中元素总的个数
print(x.numel())

# 调整张量的形状reshape
X = x.reshape(3, 4)
# X = x.reshape(-1, 4)
# X = x.reshape(3, -1)
print(X)

# 初始化为全0 或者 全1
print(torch.zeros(2, 3, 4))
print(torch.ones(2, 3, 4))

# 生成均值为0 标准差为1 标准高斯正太分布
print(torch.randn(3, 4))

# 使用列表进行一个初始化
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

