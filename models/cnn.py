import torch
#import torchvision
#import torchvision.transforms as transforms
#import numpy as np
#import matplotlib.pyplot as plt 
#import pandas as pd 


import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
#import torchvision.transforms as T

class A_Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(A_Net, self).__init__()
        self.conv1 = nn.Conv2d(input_dim,64,15,3)
        self.conv1_bn=nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,128,5,2)
        self.conv2_bn=nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,128,3,2, padding= 1)
        self.conv3_bn=nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,256,3,2, padding= 1)
        self.conv4_bn=nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256,256,3,2, padding= 1)
        self.conv5_bn=nn.BatchNorm2d(256)
        #self.conv6 = nn.Conv2d(256,256,3,1, padding= 1)
        #self.conv6_bn=nn.BatchNorm2d(256)
        #self.conv7 = nn.Conv2d(256,256,3,1, padding= 1)
        #self.conv7_bn=nn.BatchNorm2d(256)
        #self.flat = nn.Conv2d(256,512,7,1)
        self.dropout = nn.Dropout(0.1)
        #self.fc1 = nn.Linear(2048, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, 250)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self,x):
        #transform = T.Resize(size = (225,225))
        #x = transform(x)
        
        x = F.interpolate(x, (225, 225), mode='bilinear')
        
        x = F.relu(self.conv1_bn(self.conv1(x)))
        #print(x.shape)
        x = F.max_pool2d(x, 3, 2)
        
        #print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))
        
        #print(x.shape)
        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print(x.shape)
        #x = F.max_pool2d(F.relu(x), 3, 2)
        #x2=F.max_pool2d(F.relu(x), 3, 2)
        #x = torch.cat((x1,x2), 1)
        #print(x.shape)
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        #x = F.relu(self.conv6_bn(self.conv6(x)))
        #x = F.relu(self.conv7_bn(self.conv7(x)))
        #print(x.shape)
        
        #x = F.max_pool2d(F.relu(x), 3, 2)
        #x2=F.max_pool2d(F.relu(x), 3, 2)
        
        #x = torch.cat((x1,x2), 1)
        #print(x.shape)
        #x = F.relu(self.flat(x))
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        #assert False, x.shape
        #print(x.shape)
        return torch.sigmoid(self.fc(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class Bottleneck(nn.Module):
    def __init__(self, 
        in_channels, 
        out_channels, 
        kernel_size=1, 
        stride=1,
        padding=0,
        dilation=1, 
        downsample=None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 =  nn.Conv2d(
            out_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=padding,
            #dilation=dilation, 
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #return out 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
                
        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class AS_Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1,64,7,padding=3, stride=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.block1 = self._make_layer(
            2,
            64,
            128,
            3,
            2,
            1,
            0
        )
        self.block2 = self._make_layer(
            2,
            128,
            256,
            3,
            2,
            1,
            0
        )
        self.block3 = self._make_layer(
            2,
            256,
            512,
            3,
            2,
            1,
            0
        )

        self.fconv1 = nn.Conv2d(1,64,7,padding=3, stride=2)
        self.fconv1_bn=nn.BatchNorm2d(64)
        self.fblock1 = self._make_layer(
            2,
            64,
            128,
            3,
            2,
            1,
            0
        )
        self.fblock2 = self._make_layer(
            2,
            128,
            256,
            3,
            2,
            1,
            0
        )
        self.fblock3 = self._make_layer(
            2,
            256,
            512,
            3,
            2,
            1,
            0
        )

        #self.conv6 = nn.Conv2d(256,256,3,1, padding= 1)
        #self.conv6_bn=nn.BatchNorm2d(256)
        #self.conv7 = nn.Conv2d(256,256,3,1, padding= 1)
        #self.conv7_bn=nn.BatchNorm2d(256)
        #self.flat = nn.Conv2d(256,512,7,1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(256, 250)
        #self.fc2 = nn.Linear(512, num_classes)
    def _make_layer(self, 
        num_layer,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding, 
        dilation):
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        #return downsample
        layers = []
        layers.append(Bottleneck(in_channels, out_channels, kernel_size,
            stride, padding, dilation, downsample))
        for i in range(1, num_layer):
            layers.append(Bottleneck(out_channels, out_channels))
        return nn.Sequential(*layers)    
    def forward(self,data):
        x, x_ = data[:, :1, ...], data[:, 1:, ...]
        
        # A
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.max_pool2d(x, 3, 2)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        # S
        x_ = F.relu(self.fconv1_bn(self.fconv1(x_)))
        x_ = F.max_pool2d(x_, 3, 2)
        x_ = self.fblock1(x_)
        x_ = self.fblock2(x_)
        x_ = self.fblock3(x_)
        x_ = self.avgpool(x_)
        x_ = torch.flatten(x_, 1)
        #assert False, (x.shape, x_.shape)
        return self.fc(torch.cat((x, x_), dim=-1))
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = torch.sigmoid(x)
        #return x
'''
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
'''
def create_cnn(input_dim, num_classes):
    model = A_Net(input_dim, num_classes)
    return model

def create_AS(input_dim, num_classes):
    model = AS_Net(input_dim, num_classes)
    return model

if __name__ == "__main__":
    device = torch.device("cuda")
    model = create_AS(2, 250).to(device)
    fake = torch.rand(64, 2, 256, 256).float().to(device)
    pred = model(fake)
    print(pred.shape)