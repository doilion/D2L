import math
import time
import numpy as np
import torch
from d2l import torch as d2l

# 这个n值需要大一点，否则矢量化的时候会显示为0
n = 1000000
a = torch.ones([n])
b = torch.ones([n])

class Timer:
    """记录多次运行的时间"""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """开始启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """返回时间总和"""
        return sum(self.times)

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

# 逐元素加法测试
c = torch.zeros(n)
timer1 = Timer()
for i in range(n):
    c[i] = a[i] + b[i]
print(f'{timer1.stop():.8f} sec')

# 矢量化加法测试
timer2 = Timer()
d = a + b
print(f'{timer2.stop():.8f} sec')