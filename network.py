import torch
import segmentation_models_pytorch as smp

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model = smp.UnetPlusPlus(
        encoder_name = 'resnet50',
        encoder_weights = 'imagenet',
        in_channels = 3,
        classes = 1,
        activation = 'sigmoid')
        
    def forward(self, x):
        y = self.model(x)
        return y
