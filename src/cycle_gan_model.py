# src/cycle_gan_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim)

    def build_conv_block(self, dim):
        conv_block = []
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
                       nn.InstanceNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=9):
        super(Generator, self).__init__()
        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=False),
                 nn.InstanceNorm2d(ngf),
                 nn.ReLU(True)]

        for i in range(2):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=False),
                      nn.InstanceNorm2d(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**2
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult)]

        for i in range(2):
            mult = 2**(2 - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=False),
                      nn.InstanceNorm2d(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

def load_generator_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model = Generator().to(device)
    # Adjust for nested module
    state_dict = {k.replace('model.', ''): v for k, v in checkpoint['genA_state_dict'].items()}
    model.load_state_dict(state_dict)
    return model
