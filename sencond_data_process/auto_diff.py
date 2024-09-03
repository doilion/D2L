import torch

x = torch.arange(4.0)
x.requires_grad_(True)

y = 2 * torch.dot(x, x)

# 开始进行反向传播
y.backward()
print(x.grad)
print(x.grad == 4 * x)


# 在默认情况下,PyTorch会累积梯度,我们需要清除之前的值 防止梯度累计
x.grad.zero_()
y = x.sum()  
y.backward()
print(x.grad)

