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

class A_Net(nn.Module):
    def __init__(self):
        super(A_Net, self).__init__()
        self.conv1 = nn.Conv2d(1,64,15,3)
        
        self.conv2 = nn.Conv2d(64,128,5,1)
        self.conv3 = nn.Conv2d(128,128,3,1, padding= 1)
        self.conv4 = nn.Conv2d(256,256,3,1, padding= 1)
        self.conv5 = nn.Conv2d(256,256,3,1, padding= 1)
        self.conv6 = nn.Conv2d(256,256,3,1, padding= 1)
        self.conv7 = nn.Conv2d(256,256,3,1, padding= 1)
        self.flat = nn.Conv2d(512,512,7,1)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 4096)
        self.fc3 = nn.Linear(4096, 250)

    def forward(self,x):
        transform = T.Resize(size = (225,225))
        x = transform(x)
        x=self.conv1(x)
        print(x.shape)
        x=F.max_pool2d(F.relu(x), 3, 2)
        print(x.shape)
        x=F.relu(self.conv2(x))
        
        print(x.shape)
        x=self.conv3(x)
        print(x.shape)
        x1=F.max_pool2d(F.relu(x), 3, 2)
        x2=F.max_pool2d(F.relu(x), 3, 2)
        x = torch.cat((x1,x2), 1)
        print(x.shape)
        x=F.relu(self.conv4(x))
        x=F.relu(self.conv5(x))
        x=F.relu(self.conv6(x))
        x=F.relu(self.conv7(x))
        print(x.shape)
        x1=F.max_pool2d(F.relu(x), 3, 2)
        x2=F.max_pool2d(F.relu(x), 3, 2)
        
        x = torch.cat((x1,x2), 1)
        print(x.shape)
        x = F.relu(self.flat(x))
        print(x.shape)
        x = torch.flatten(x,1)
        print(x.shape)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionalUnit(nn.Module):
    def __init__(self, structure_type):
        super(ConvolutionalUnit, self).__init__()
        self.structure_type = structure_type

        if structure_type == 'classic':
            self.net = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )
        elif structure_type == 'advanced':
            self.net = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
            )
        else:
            raise ValueError(structure_type)

    def forward(self, x):
        residual = x
        x = self.net(x)
        if self.structure_type == 'advanced':
            x = 0.1 * x
        x = residual + x
        return x

def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)
    
    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]
        
        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

class S_Net(nn.Module):
    def __init__(self, num_metrics=8, structure_type='classic'):
        super(S_Net, self).__init__()
        self.num_metrics = num_metrics

        self.encoder = EncoderBlock()
        self.convolution_units = nn.Sequential(*[ConvolutionalUnit(structure_type) for i in range(num_metrics)])
        self.decoders = nn.Sequential(*[DecoderBlock() for i in range(num_metrics)])

    def forward(self, x):
        x = fps(x)

        outs = []
        prev_out = x
        for i in range(self.num_metrics):
            out = self.convolution_units[i](prev_out)
            prev_out = out
            outs.append(self.decoders[i](out))

        return outs

def create_model(input_dim, num_classes):
    model = A_Net()
    return model


if __name__ == "__main__":
    device = torch.device("cuda")
    model = create_model(2, 250).to(device)
    fake = torch.rand(64, 2, 256, 256).to(device)
    pred = model(fake)
    print(pred.shape)