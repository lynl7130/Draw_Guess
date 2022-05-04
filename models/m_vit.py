from vit_pytorch import ViT
import torch.nn as nn
class A_Net(nn.Module):
    def __init__(self, num_classes):
        super(A_Net, self).__init__()
        self.v = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 250,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            channels = 1,
            emb_dropout = 0.1
        )

    def forward(self,x):
        x=self.v(x)
        return x

def create_model(num_classes):
    model = A_Net(num_classes)
    return model
