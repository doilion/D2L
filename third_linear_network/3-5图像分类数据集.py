import torch
import matplotlib.pyplot as plt
from torch.utils import data
from torchvision import datasets, transforms
from d2l import torch as d2l

d2l.use_svg_display()

trans = transforms.ToTensor()
mnist_train = datasets.FashionMNIST
