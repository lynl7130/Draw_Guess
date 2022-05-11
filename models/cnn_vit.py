import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from vit_pytorch import ViT

class A_Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(A_Net, self).__init__()
        
        self.conv1  = nn.Conv2d(input_dim,8,7,3)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8,16,3,1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,16,3,1)
        self.bn3 = nn.BatchNorm2d(16)
        self.v = ViT(
            image_size = 80,
            patch_size = 20,
            num_classes = num_classes,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            channels = 16,
            emb_dropout = 0.1
        )

    def forward(self,x):
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=F.relu(self.bn3(self.conv3(x)))
        x=self.v(x)
        return x

def create_model(input_dim, num_classes):
    model = A_Net(input_dim, num_classes)
    return model


if __name__ == "__main__":
    device = torch.device("cuda")
    model = create_model(250).to(device)
    fake = torch.rand(64, 2, 256, 256).to(device)
    pred = model(fake)
    print(pred.shape)