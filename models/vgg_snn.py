import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
from snntorch import functional as SF
from snntorch import spikeplot as splt


class VGG16(nn.Module):
    def __init__(self, spike_grad = surrogate.atan(), beta = 0.9, num_classes=1000):
        super(VGG16, self).__init__()
        self.spike_grad = spike_grad
        self.beta = beta
        self.num_classes = num_classes
        self.features()
        self.classifier()

    def features(self):
        self.conv1 = nn.Conv2d(4, 64, kernel_size=3, padding=1)
        beta1 = torch.rand(64)
        self.lif1 = snn.Leaky(beta=beta1, learn_beta=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        beta2 = torch.rand(64)
        self.lif2 = snn.Leaky(beta=beta2, learn_beta=True)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        beta3 = torch.rand(128)
        self.lif3 =snn.Leaky(beta=beta3, learn_beta=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        beta4 = torch.rand(128)
        self.lif4 = snn.Leaky(beta=beta4, learn_beta=True)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        beta5 = torch.rand(256)
        self.lif5 = snn.Leaky(beta=beta5, learn_beta=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        beta6 = torch.rand(256)
        self.lif6 = snn.Leaky(beta=beta6, learn_beta=True)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        beta7 = torch.rand(256)
        self.lif7 = snn.Leaky(beta=beta7, learn_beta=True)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        beta8 = torch.rand(512)
        self.lif8 = snn.Leaky(beta=beta8, learn_beta=True)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        beta9 = torch.rand(512)
        self.lif9 = snn.Leaky(beta=beta9, learn_beta=True)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        beta10 = torch.rand(512)
        self.lif10 = snn.Leaky(beta=beta10, learn_beta=True)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        beta11 = torch.rand(512)
        self.lif11 = snn.Leaky(beta=beta11, learn_beta=True)
        self.conv12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        beta12 = torch.rand(512)
        self.lif12 = snn.Leaky(beta=beta12, learn_beta=True)
        self.conv13 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        beta13 = torch.rand(512)
        self.lif13 = snn.Leaky(beta=beta13, learn_beta=True)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        beta14 = torch.rand(512)
        self.lif14 = snn.Leaky(beta=beta14, learn_beta=True)
        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

    def classifier(self):
        self.flt1 = nn.Flatten()
        self.lin1 = nn.Linear(512 * 2 * 2, 256)
        beta15 = torch.rand(256)
        self.lif15 = snn.Leaky(beta=beta15, learn_beta=True)
        self.drp1 = nn.Dropout()
        self.lin2 = nn.Linear(256, 64)
        beta16 = torch.rand(64)
        self.lif16 = snn.Leaky(beta=beta16, learn_beta=True)
        self.drp2 = nn.Dropout()
        self.lin3 = nn.Linear(64, self.num_classes)
        beta17 = torch.rand(2)
        self.lif17 = snn.Leaky(beta=beta17, learn_beta=True, spike_grad=self.spike_grad, output=True)


    def forward_features(self, x):
        x1 = self.conv1(x)
        x2, self.mem1 = self.lif1(x1, self.mem1)
        x3 = self.conv2(x2)
        x4, self.mem2 = self.lif2(x3, self.mem2)
        x5 = self.mp1(x4)
        x6 = self.conv3(x5)
        x7, self.mem3 = self.lif3(x6, self.mem3)
        x8 = self.conv4(x7)
        x9, self.mem4 = self.lif4(x8, self.mem4)
        x10 = self.mp2(x9)
        x11 = self.conv5(x10)
        x12, self.mem5 = self.lif5(x11, self.mem5)
        x13 = self.conv6(x12)
        x14, self.mem6 = self.lif6(x13, self.mem6)
        x15 = self.conv7(x14)
        x16, self.mem7 = self.lif7(x15, self.mem7)
        x17 = self.mp3(x16)
        x18 = self.conv8(x17)
        x19, self.mem8 = self.lif8(x18, self.mem8)
        x20 = self.conv9(x19)
        x21, self.mem9 = self.lif9(x20, self.mem9)
        x22 = self.conv10(x21) 
        x23, self.mem10 = self.lif10(x22, self.mem10)
        x24 = self.mp4(x23)
        x25 = self.conv11(x24) 
        x26, self.mem11 = self.lif11(x25, self.mem11)
        x27 = self.conv12(x26) 
        x28, self.mem12 = self.lif12(x27, self.mem12)
        x29 = self.conv13(x28) 
        x30, self.mem13 = self.lif13(x29, self.mem13)
        x31 = self.mp5(x30)
        x32, self.mem14 = self.lif14(x31, self.mem14)

        return x32


    def forward_classifier(self, x):
        x32 = self.flt1(x)
        x33 = self.lin1(x32) 
        x34, self.mem15 = self.lif15(x33, self.mem15)
        x35 = self.drp1(x34) 
        x36 = self.lin2(x35)
        x37, self.mem16 = self.lif16(x36, self.mem16)
        x38 = self.drp2(x37) 
        x39 = self.lin3(x38)
        x40, self.mem17 = self.lif17(x39, self.mem17) 

        return x40, self.mem17


    def init_leaky(self):
        self.mem1 = self.lif1.init_leaky()
        self.mem2 = self.lif2.init_leaky()
        self.mem3 = self.lif3.init_leaky()
        self.mem4 = self.lif4.init_leaky()
        self.mem5 = self.lif5.init_leaky()
        self.mem6 = self.lif6.init_leaky()
        self.mem7 = self.lif7.init_leaky()
        self.mem8 = self.lif8.init_leaky()
        self.mem9 = self.lif9.init_leaky()
        self.mem10 = self.lif10.init_leaky()
        self.mem11 = self.lif11.init_leaky()
        self.mem12 = self.lif12.init_leaky()
        self.mem13 = self.lif13.init_leaky()
        self.mem14 = self.lif14.init_leaky()
        self.mem15 = self.lif15.init_leaky()
        self.mem16 = self.lif16.init_leaky()
        self.mem17 = self.lif17.init_leaky()


    def forward(self, x1):
        spk_rec = []
        self.init_leaky()
        for step in range(x1.size(0)):  # data.size(0) = number of time steps
            x = self.forward_features(x1[step])
            spk_out, __ = self.forward_classifier(x)
            spk_rec.append(spk_out)
        return torch.stack(spk_rec)
        