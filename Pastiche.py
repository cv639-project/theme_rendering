"""model to be trained to apply filters"""
import torch
import torch.nn as nn

class PasticheModel(torch.nn.Module):
    def __init__(self, num_themes):
        super(PasticheModel, self).__init__()

        self.conv_layer1 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(3, 32, (9, 9))
        )
        self.conv_layer2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 64, (3, 3), stride=2)
        )
        self.conv_layer3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, (3, 3), stride=2)
        )

        self.conv_layer4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, (3, 3))
        )

        self.conv_layer5 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, (3, 3))
        )
        self.conv_layer6 = nn.Sequential(
            nn.ReflectionPad2d(4),
            nn.Conv2d(32, 3, (9, 9))
        )


        self.In1 = Cond_InN(num_themes, 32)
        self.In2 = Cond_InN(num_themes, 64)
        self.In3 = Cond_InN(num_themes, 128)
        self.In4 = Cond_InN(num_themes, 64)
        self.In5 = Cond_InN(num_themes, 32)

        self.residual_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, (3, 3)),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, (3, 3)),
            nn.InstanceNorm2d(128, affine=True)
        )

    def forward(self, x, theme_ids):
        y = self.conv_layer1(x)
        y = self.In1(y, theme_ids)

        y = self.conv_layer2(y)
        y = self.In2(y, theme_ids)

        y = self.conv_layer3(y)
        y = self.In3(y, theme_ids)

        y = y + self.residual_block(y)
        y = y + self.residual_block(y)
        y = y + self.residual_block(y)
        y = y + self.residual_block(y)

        y = self.conv_layer4(y)
        y = self.In4(y, theme_ids)

        y = self.conv_layer5(y)
        y = self.In5(y, theme_ids)

        y = self.conv_layer6(y)

        return y

class Cond_InN(nn.Module):
    def __init__(self, num_themes, output_filters):
        super(Cond_InN, self).__init__()

        self.relu = nn.ReLU()
        self.InN = nn.InstanceNorm2d(output_filters, affine=True)
        # initialize gamma matrix with 1s
        self.gamma = torch.nn.Parameter(data=torch.Tensor(num_themes, output_filters))
        self.gamma.data.uniform_(1, 1)
        # initialize beta matrix with random numbers from 0-1
        self.beta = torch.nn.Parameter(data=torch.Tensor(num_themes, output_filters))
        self.beta.data.uniform_(0, 1)

    def forward(self, x, theme_ids):
        y = self.InN(x)
        b, d, w, h = y.size()
        y = y.view(b, d, w * h)
        # cell expansion
        gamma = self.gamma[theme_ids]
        beta = self.beta[theme_ids]
        y = (y * gamma.unsqueeze(-1).expand_as(y) + beta.unsqueeze(-1).expand_as(y)).view(b, d, w, h)
        y = self.relu(y)

        return y
