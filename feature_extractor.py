"""feature extraction"""
import torch
from torchvision import models
"""
vgg11 features:

Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (4): ReLU(inplace=True)
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (6): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU(inplace=True)
  (8): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (9): ReLU(inplace=True)
  (10): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (11): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (12): ReLU(inplace=True)
  (13): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (14): ReLU(inplace=True)
  (15): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  
  (16): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (17): ReLU(inplace=True)
  (18): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (19): ReLU(inplace=True)
  (20): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
"""

class vgg11(torch.nn.Module):
    def __init__(self):
        super(vgg11, self).__init__()
        vgg_pretrained_features = models.vgg11(pretrained=True).features

        self.block1 = torch.nn.Sequential()
        self.block2 = torch.nn.Sequential()
        self.block3 = torch.nn.Sequential()
        self.block4 = torch.nn.Sequential()
        self.block5 = torch.nn.Sequential()
        
        for i in range(0, 2):
            self.block1.add_module(str(i), vgg_pretrained_features[i])
        for i in range(2, 5):
            self.block2.add_module(str(i), vgg_pretrained_features[i])
        for i in range(5, 10):
            self.block3.add_module(str(i), vgg_pretrained_features[i])
        for i in range(10, 15):
            self.block4.add_module(str(i), vgg_pretrained_features[i])
        for i in range(15, 20):
            self.block5.add_module(str(i), vgg_pretrained_features[i])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, X):
        # individual out1 to out5 for theme features (low levels to high levels)
        out1 = self.block1(X)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)

        out = [out1, out2, out3, out4, out5]
        # level of details of filters can be changed
        # through block selection
        # out = out
        out = out[1:]
        # out = out[2:]

        return out
