import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, beta_shape, spike_grad = surrogate.atan()):
        super(DepthwiseSeparableConv, self).__init__()
        self.spike_grad = spike_grad
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.beta1 = torch.rand(beta_shape)
        self.thr1 = torch.rand(beta_shape)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.beta2 = torch.rand(beta_shape)
        self.thr2 = torch.rand(beta_shape)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)

    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def forward(self, x):
        self.init_leaky()
        out2, self.mem1 = self.lif1(self.bn1(self.depthwise(x)), self.mem1)
        out4, self.mem2 = self.lif2(self.bn2(self.pointwise(out2)), self.mem2)
        return out4

class MobileNetV1(nn.Module):
    def __init__(self, init_hidden_layers=8, num_classes=2, spike_grad = surrogate.atan()):
        super(MobileNetV1, self).__init__()
        self.spike_grad = spike_grad
        self.conv1 = nn.Conv2d(4, init_hidden_layers, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(init_hidden_layers)
        self.beta1 = torch.rand(32)
        self.thr1 = torch.rand(32)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.dsconv1 = DepthwiseSeparableConv(init_hidden_layers, 2*init_hidden_layers, stride=1, beta_shape=32)
        self.dsconv2 = DepthwiseSeparableConv(2*init_hidden_layers, 4*init_hidden_layers, stride=2, beta_shape=16)
        self.dsconv3 = DepthwiseSeparableConv(4*init_hidden_layers, 4*init_hidden_layers, stride=1, beta_shape=16)
        self.dsconv4 = DepthwiseSeparableConv(4*init_hidden_layers, 8*init_hidden_layers, stride=2, beta_shape=8)
        self.dsconv5 = DepthwiseSeparableConv(8*init_hidden_layers, 8*init_hidden_layers, stride=1, beta_shape=8)
        self.dsconv6 = DepthwiseSeparableConv(8*init_hidden_layers, 16*init_hidden_layers, stride=2, beta_shape=4)
        self.dsconv7 = DepthwiseSeparableConv(16*init_hidden_layers, 16*init_hidden_layers, stride=1, beta_shape=4)
        self.dsconv8 = DepthwiseSeparableConv(16*init_hidden_layers, 16*init_hidden_layers, stride=1, beta_shape=4)
        self.dsconv9 = DepthwiseSeparableConv(16*init_hidden_layers, 16*init_hidden_layers, stride=1, beta_shape=4)
        self.dsconv10 = DepthwiseSeparableConv(16*init_hidden_layers, 32*init_hidden_layers, stride=2, beta_shape=2)
        self.dsconv11 = DepthwiseSeparableConv(32*init_hidden_layers, 32*init_hidden_layers, stride=1, beta_shape=2)
        self.flt = nn.Flatten()
        self.fc1 = nn.Linear(32*init_hidden_layers*4, 128)
        self.beta2 = torch.rand(128)
        self.thr2 = torch.rand(128)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.fc2 = nn.Linear(128, num_classes)
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
            out, self.mem1 = self.lif1(self.bn1(self.conv1(x1[step])), self.mem1)                 
            out = self.dsconv1(out)                                               # [64, 64, 32, 32]
            out = self.dsconv2(out)                                               # [64, 128, 16, 16]
            out = self.dsconv3(out)                                               # [64, 128, 16, 16]
            out = self.dsconv4(out)                                               # [64, 256, 8, 8]
            out = self.dsconv5(out)                                               # [64, 256, 8, 8]
            out = self.dsconv6(out)                                               # [64, 512, 4, 4]
            out = self.dsconv7(out)                                               # [64, 512, 4, 4]
            out = self.dsconv8(out)                                              # [64, 512, 4, 4]
            out = self.dsconv9(out)                                             # [64, 512, 4, 4]
            out = self.dsconv10(out)                                            # [64, 1024, 2, 2]
            out = self.dsconv11(out)                                            # [64, 1024, 2, 2]
            out = self.flt(out)                                            # [64, 1024]
            out, self.mem2 = self.lif2(self.fc1(out), self.mem2)                   # [64, 2]
            out, self.mem3 = self.lif3(self.fc2(out), self.mem3)
            spk_rec.append(out)
        return torch.stack(spk_rec)
