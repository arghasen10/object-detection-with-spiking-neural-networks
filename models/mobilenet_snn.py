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
    def __init__(self, num_classes=2, spike_grad = surrogate.atan()):
        super(MobileNetV1, self).__init__()
        self.spike_grad = spike_grad
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.beta1 = torch.rand(32)
        self.thr1 = torch.rand(32)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.dsconv1 = DepthwiseSeparableConv(32, 64, stride=1, beta_shape=32)
        self.dsconv2 = DepthwiseSeparableConv(64, 128, stride=2, beta_shape=16)
        self.dsconv3 = DepthwiseSeparableConv(128, 128, stride=1, beta_shape=16)
        self.dsconv4 = DepthwiseSeparableConv(128, 256, stride=2, beta_shape=8)
        self.dsconv5 = DepthwiseSeparableConv(256, 256, stride=1, beta_shape=8)
        self.dsconv6 = DepthwiseSeparableConv(256, 512, stride=2, beta_shape=4)
        self.dsconv7 = DepthwiseSeparableConv(512, 512, stride=1, beta_shape=4)
        self.dsconv8 = DepthwiseSeparableConv(512, 512, stride=1, beta_shape=4)
        self.dsconv9 = DepthwiseSeparableConv(512, 512, stride=1, beta_shape=4)
        self.dsconv10 = DepthwiseSeparableConv(512, 1024, stride=2, beta_shape=2)
        self.dsconv11 = DepthwiseSeparableConv(1024, 1024, stride=1, beta_shape=2)
        self.flt = nn.Flatten()
        self.fc = nn.Linear(4096, num_classes)
        self.beta2 = torch.rand(num_classes)
        self.thr2 = torch.rand(num_classes)
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)

    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def forward(self, x1):
        spk_rec = []
        self.init_leaky()
        for step in range(x1.size(0)):  # data.size(0) = number of time steps
            out2, self.mem1 = self.lif1(self.bn1(self.conv1(x1[step])), self.mem1)                 
            out3 = self.dsconv1(out2)                                               # [64, 64, 32, 32]
            out4 = self.dsconv2(out3)                                               # [64, 128, 16, 16]
            out5 = self.dsconv3(out4)                                               # [64, 128, 16, 16]
            out6 = self.dsconv4(out5)                                               # [64, 256, 8, 8]
            out7 = self.dsconv5(out6)                                               # [64, 256, 8, 8]
            out8 = self.dsconv6(out7)                                               # [64, 512, 4, 4]
            out9 = self.dsconv7(out8)                                               # [64, 512, 4, 4]
            out10 = self.dsconv8(out9)                                              # [64, 512, 4, 4]
            out11 = self.dsconv9(out10)                                             # [64, 512, 4, 4]
            out12 = self.dsconv10(out11)                                            # [64, 1024, 2, 2]
            out13 = self.dsconv11(out12)                                            # [64, 1024, 2, 2]
            out15 = self.flt(out13)                                            # [64, 1024]
            out, self.mem2 = self.lif2(self.fc(out15), self.mem2)                   # [64, 2]
            
            spk_rec.append(out)
        return torch.stack(spk_rec)
