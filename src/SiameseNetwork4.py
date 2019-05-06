import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Code taken from
# https://discuss.pytorch.org/t/flatten-layer-of-pytorch-build-by-sequential-container/5983
# https://discuss.pytorch.org/t/converting-2d-to-1d-module/25796/3
class Flatten(torch.nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class SiameseNetwork4(torch.nn.Module):
    
    def __init__(self):
        
        super(SiameseNetwork4, self).__init__()
        self.model    = models.resnet50(pretrained=True)
        print("Using Siamese Network of Resnet 50")
        self.model.avgpool = nn.MaxPool2d(kernel_size=12)
        self.model.fc = Flatten()

    def forward_once(self, x):
        x = self.model(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
