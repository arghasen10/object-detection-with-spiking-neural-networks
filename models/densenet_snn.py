import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

# Define the Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, spike_grad = surrogate.atan()):
        super(DenseBlock, self).__init__()
        self.spike_grad = spike_grad
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1)
        beta1 = torch.rand(int(growth_rate/2))
        thr1 = torch.rand(int(growth_rate/2))
        self.lif1 = snn.Leaky(beta=beta1, threshold=thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1)
        beta2 = torch.rand(int(growth_rate/2))
        thr2 = torch.rand(int(growth_rate/2))
        self.lif2 = snn.Leaky(beta=beta2, threshold=thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
    
    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def forward(self, x):
        self.init_leaky()
        out1, self.mem1 = self.lif1(self.conv1(x), self.mem1)
        out2, self.mem2 = self.lif2(self.conv2(out1), self.mem2)
        out = torch.cat((x, out2), 1)
        return out

# Define the Transition Block
class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spike_grad = surrogate.atan()):
        super(TransitionBlock, self).__init__()
        self.spike_grad = spike_grad
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.beta1 = torch.rand(out_channels)
        self.thr1 = torch.rand(out_channels)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.beta2 = torch.rand(out_channels)
        self.thr2 = torch.rand(out_channels)
        self.lif2 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)

    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()

    def forward(self, x):
        out1, self.mem1 = self.lif1(self.conv(x), self.mem1)
        out, self.mem2 = self.lif2(self.pool(out1), self.mem2) 
        return out

# Define the DenseNet
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_classes=2, spike_grad = surrogate.atan()):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.spike_grad = spike_grad
        # Initial convolution
        self.conv1 = nn.Conv2d(4, 2 * growth_rate, kernel_size=7, stride=2, padding=3)
        self.beta1 = torch.rand(growth_rate)
        self.thr1 = torch.rand(growth_rate)
        self.lif1 = snn.Leaky(beta=self.beta1, threshold=self.thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.beta2 = torch.rand(int(growth_rate/2))
        self.thr2 = torch.rand(int(growth_rate/2))
        self.lif2 = snn.Leaky(beta=self.beta2, threshold=self.thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        # Dense Blocks
        self.dense1 = self._make_dense_block(2 * growth_rate, block_config[0])
        self.trans1 = self._make_transition_block(2 * growth_rate)
        self.dense2 = self._make_dense_block(2 * growth_rate, block_config[1])
        self.trans2 = self._make_transition_block(2 * growth_rate)
        self.dense3 = self._make_dense_block(2 * growth_rate, block_config[2])
        self.trans3 = self._make_transition_block(2 * growth_rate)
        self.dense4 = self._make_dense_block(2 * growth_rate, block_config[3])
        
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.beta3 = torch.rand(2 * growth_rate)
        self.thr3 = torch.rand(2 * growth_rate)
        self.lif3 = snn.Leaky(beta=self.beta3, threshold=self.thr3, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.fc = nn.Linear(2 * growth_rate, num_classes)
        self.beta4 = torch.rand(2 * growth_rate)
        self.thr4 = torch.rand(2 * growth_rate)
        self.lif4 = snn.Leaky(beta=self.beta4, threshold=self.thr4, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad, output=True)
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
        
    def _make_dense_block(self, in_channels, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(DenseBlock(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)
    
    def _make_transition_block(self, in_channels):
        return TransitionBlock(in_channels, in_channels // 2)
    
    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem4 = self.lif4.init_leaky()

    def forward(self, x1):
        spk_rec = []
        self.init_leaky()
        for step in range(x1.size(0)):  # data.size(0) = number of time steps
            print('#### x shape: ', x1[step].shape)
            out, self.mem1 = self.lif1(self.conv1(x1[step]), self.mem1)
            print(' out, self.mem1 = self.lif1(self.conv1(x1[step]), self.mem1)', out.shape)
            out, self.mem2 = self.lif2(self.pool(out), self.mem2)
            out = self.dense1(out)
            out = self.trans1(out)
            out = self.dense2(out)
            out = self.trans2(out)
            out = self.dense3(out)
            out = self.trans3(out)
            out = self.dense4(out)
            out, self.mem3 = self.lif3(self.avgpool(out), self.mem3) 
            out = out.view(out.size(0), -1)
            spk_out, self.mem4 = self.lif4(self.fc(out), self.mem4)
            spk_rec.append(spk_out)
        return torch.stack(spk_rec)
        
