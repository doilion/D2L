import torch
import matplotlib.pyplot as plt

torch.manual_seed(100)

# 生成x坐标数据
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
# x = torch.linspace(-1, 1, 100)

# 生成 y 坐标数据，形成为 100 * 1， 加上一些噪声
y = 3 * x.pow(2) + 2 + 0.2 * torch.rand(x.size())

# # 把tensor数据转换为numpy数据，并可视化
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

# 初始化对应需要优化的参数
w = torch.randn(1, 1, requires_grad=True)
b = torch.zeros(1, 1, requires_grad=True)
print(w, b)

# 定义学习率
lr = 0.005

# 开始训练
for i in range(1000):
    y_pred = w * x.pow(2) + b
    loss = (y_pred - y).pow(2)

    # 反向传播 自动计算梯度
    loss.sum().backward()

    # 手动为参数进行更新，不要进行自动求导
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        # 更新一次之后 对梯度进行清零
        w.grad.zero_()
        b.grad.zero_()

print(w)
print(b)

# 进行可视化操作
plt.plot(x.numpy(), y_pred.detach().numpy(),'r-',label='predict')#predict
plt.scatter(x.numpy(), y.numpy(),color='blue',marker='o',label='true') # true data
plt.xlim(-1,1)
plt.ylim(2,6)
plt.legend()
plt.show()

