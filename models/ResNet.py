import torch
import torch.nn as nn
import torchvision

def create_model(input_dim, num_classes):
    model = torchvision.models.resnet18(pretrained=False,
    num_classes=num_classes)
    model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

if __name__ == "__main__":
    device = torch.device("cuda")
    model = create_model(2, 250).to(device)
    fake = torch.rand(64, 2, 256, 256).to(device)
    pred = model(fake)
    print(pred.shape)