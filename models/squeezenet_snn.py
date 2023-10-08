import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class FireModule(nn.Module):
    def __init__(self, in_channels,  s1x1, e1x1, e3x3, spike_grad = surrogate.atan(), beta_shape=14):
        super(FireModule, self).__init__()
        self.spike_grad = spike_grad
        self.squeeze = nn.Conv2d(in_channels, s1x1, kernel_size=1)
        self.beta1 = torch.rand(beta_shape)
        self.thr1 = torch.rand(beta_shape)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.expand1x1 = nn.Conv2d(s1x1, e1x1, kernel_size=1)
        self.beta2 = torch.rand(beta_shape)
        self.thr2 = torch.rand(beta_shape)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.expand3x3 = nn.Conv2d(s1x1, e3x3, kernel_size=3, padding=1)
        self.beta3 = torch.rand(beta_shape)
        self.thr3 = torch.rand(beta_shape)
        self.lif3 = snn.Leaky(beta=self.beta3, threshold=self.thr3, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
    
    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()

    def forward(self, x):
        self.init_leaky()
        squeezed, self.mem1 = self.lif1(self.squeeze(x), self.mem1)
        expanded1x1, self.mem2 = self.lif2(self.expand1x1(squeezed), self.mem2)
        expanded3x3, self.mem3 = self.lif3(self.expand3x3(squeezed), self.mem3)
        return torch.cat([expanded1x1, expanded3x3], 1)

class SqueezeNet(nn.Module):
    def __init__(self, beta_shape=29, spike_grad=surrogate.atan(), num_classes=2):
        super(SqueezeNet, self).__init__()
        self.spike_grad = spike_grad
        self.conv1 = nn.Conv2d(4, 96, kernel_size=7, stride=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.beta1 = torch.rand(beta_shape)
        self.thr1 = torch.rand(beta_shape)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.fire2 = FireModule(96, 16, 64, 64)
        self.fire3 = FireModule(128, 16, 64, 64)
        self.fire4 = FireModule(128, 32, 128, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire5 = FireModule(256, 32, 128, 128, beta_shape=6)
        self.fire6 = FireModule(256, 48, 192, 192, beta_shape=6)
        self.fire7 = FireModule(384, 48, 192, 192, beta_shape=6)
        self.fire8 = FireModule(384, 64, 256, 256, beta_shape=6)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fire9 = FireModule(512, 64, 256, 256, beta_shape=2)
        self.conv10 = nn.Conv2d(512, 64, kernel_size=1)
        self.beta2 = torch.rand(2)
        self.thr2 = torch.rand(2)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad, output=True)
        self.flt = nn.Flatten()
        self.fc = nn.Linear(256, num_classes)
        self.beta3 = torch.rand(num_classes)
        self.thr3 = torch.rand(num_classes)
        self.lif3 = snn.Leaky(beta=self.beta3, threshold=self.thr3, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
    
    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()

    def forward(self, x1):
        spk_rec = []
        self.init_leaky()
        for step in range(x1.size(0)):  # data.size(0) = number of time steps
            x, self.mem1 = self.lif1(self.conv1(x1[step]), self.mem1)           # I/P [64, 4, 64, 64], O/P [64, 96, 29, 29]
            x = self.maxpool1(x)                                                # [64, 96, 14, 14]
            x = self.fire2(x)                                                   # [64, 128, 14, 14]
            x = self.fire3(x)                                                   # [64, 128, 14, 14]
            x = self.fire4(x)                                                   # [64, 256, 14, 14]
            x = self.maxpool2(x)                                                # [64, 256, 6, 6]
            x = self.fire5(x)                                                   # [64, 256, 6, 6]
            x = self.fire6(x)                                                   # [64, 384, 6, 6]
            x = self.fire7(x)                                                   # [64, 384, 6, 6]
            x = self.fire8(x)                                                   # [64, 512, 6, 6]
            x = self.maxpool3(x)                                                # [64, 512, 2, 2]
            x = self.fire9(x)                                                   # [64, 512, 2, 2]
            x, self.mem2 = self.lif2(self.conv10(x), self.mem2)                 # [64, 64, 2, 2]
            x = self.flt(x)                                                     # [64, 256]
            x, self.mem3 = self.lif3(self.fc(x), self.mem3)                     # [64, 2]
            spk_rec.append(x)
        return torch.stack(spk_rec)

