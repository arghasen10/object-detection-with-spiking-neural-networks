import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class VGG16(nn.Module):
    def __init__(self, spike_grad = surrogate.atan(), init_hidden_layers=8, beta = 0.9, num_classes=1000):
        super(VGG16, self).__init__()
        self.spike_grad = spike_grad
        self.beta = beta
        self.num_classes = num_classes
        self.features(init_hidden_layers=init_hidden_layers)
        self.classifier(init_hidden_layers=init_hidden_layers)

    def features(self, init_hidden_layers=8):
        self.conv1 = nn.Conv2d(4, init_hidden_layers, kernel_size=3, padding=1)
        beta1 = torch.rand(64)
        thr1 = torch.rand(64)
        self.lif1 = snn.Leaky(beta=beta1, threshold=thr1, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv2 = nn.Conv2d(init_hidden_layers, init_hidden_layers, kernel_size=3, padding=1)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        beta2 = torch.rand(32)
        thr2 = torch.rand(32)
        self.lif2 = snn.Leaky(beta=beta2, threshold=thr2, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv3 = nn.Conv2d(init_hidden_layers, 2*init_hidden_layers, kernel_size=3, padding=1)
        beta3 = torch.rand(32)
        thr3 = torch.rand(32)
        self.lif3 =snn.Leaky(beta=beta3, threshold=thr3, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv4 = nn.Conv2d(2*init_hidden_layers, 2*init_hidden_layers, kernel_size=3, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        beta4 = torch.rand(16)
        thr4 = torch.rand(16)
        self.lif4 = snn.Leaky(beta=beta4, threshold=thr4, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv5 = nn.Conv2d(2*init_hidden_layers, 4*init_hidden_layers, kernel_size=3, padding=1)
        beta5 = torch.rand(16)
        thr5 = torch.rand(16)
        self.lif5 = snn.Leaky(beta=beta5, threshold=thr5, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv6 = nn.Conv2d(4*init_hidden_layers, 4*init_hidden_layers, kernel_size=3, padding=1)
        beta6 = torch.rand(16)
        thr6 = torch.rand(16)
        self.lif6 = snn.Leaky(beta=beta6, threshold=thr6, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv7 = nn.Conv2d(4*init_hidden_layers, 4*init_hidden_layers, kernel_size=3, padding=1)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        beta7 = torch.rand(8)
        thr7 = torch.rand(8)
        self.lif7 = snn.Leaky(beta=beta7, threshold=thr7, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv8 = nn.Conv2d(4*init_hidden_layers, 8*init_hidden_layers, kernel_size=3, padding=1)
        beta8 = torch.rand(8)
        thr8 = torch.rand(8)
        self.lif8 = snn.Leaky(beta=beta8, threshold=thr8, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv9 = nn.Conv2d(8*init_hidden_layers, 8*init_hidden_layers, kernel_size=3, padding=1)
        beta9 = torch.rand(8)
        thr9 = torch.rand(8)
        self.lif9 = snn.Leaky(beta=beta9, threshold=thr9, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv10 = nn.Conv2d(8*init_hidden_layers, 8*init_hidden_layers, kernel_size=3, padding=1)
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        beta10 = torch.rand(4)
        thr10 = torch.rand(4)
        self.lif10 = snn.Leaky(beta=beta10, threshold=thr10, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv11 = nn.Conv2d(8*init_hidden_layers, 8*init_hidden_layers, kernel_size=3, padding=1)
        beta11 = torch.rand(4)
        thr11 = torch.rand(4)
        self.lif11 = snn.Leaky(beta=beta11, threshold=thr11, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv12 = nn.Conv2d(8*init_hidden_layers, 8*init_hidden_layers, kernel_size=3, padding=1)
        beta12 = torch.rand(4)
        thr12 = torch.rand(4)
        self.lif12 = snn.Leaky(beta=beta12, threshold=thr12, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.conv13 = nn.Conv2d(8*init_hidden_layers, 8*init_hidden_layers, kernel_size=3, padding=1)
        self.mp5 = nn.MaxPool2d(kernel_size=2, stride=2)
        beta13 = torch.rand(2)
        thr13 = torch.rand(2)
        self.lif13 = snn.Leaky(beta=beta13, threshold=thr13, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)

    def classifier(self,init_hidden_layers=8):
        self.flt1 = nn.Flatten()
        self.lin1 = nn.Linear(8*init_hidden_layers * 2 * 2, 128)
        # self.drp1 = nn.Dropout()
        beta14 = torch.rand(128)
        thr14 = torch.rand(128)
        self.lif14 = snn.Leaky(beta=beta14, threshold=thr14, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        self.lin2 = nn.Linear(128, self.num_classes)
        # self.drp2 = nn.Dropout()
        # beta15 = torch.rand(64)
        # thr15 = torch.rand(64)
        # self.lif15 = snn.Leaky(beta=beta15, threshold=thr15, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad)
        # self.lin3 = nn.Linear(64, self.num_classes)
        beta16 = torch.rand(2)
        thr16 = torch.rand(2)
        self.lif16 = snn.Leaky(beta=beta16, threshold=thr16, learn_beta=True, learn_threshold=True, spike_grad=self.spike_grad, output=True)


    def forward_features(self, x):
        x = self.conv1(x)
        x, self.mem1 = self.lif1(x, self.mem1)
        x = self.conv2(x)
        x, self.mem2 = self.lif2(self.mp1(x), self.mem2)
        x = self.conv3(x)
        x, self.mem3 = self.lif3(x, self.mem3)
        x = self.conv4(x)
        x, self.mem4 = self.lif4(self.mp2(x), self.mem4)
        x = self.conv5(x)
        x, self.mem5 = self.lif5(x, self.mem5)
        x = self.conv6(x)
        x, self.mem6 = self.lif6(x, self.mem6)
        x = self.conv7(x)
        x, self.mem7 = self.lif7(self.mp3(x), self.mem7)
        x = self.conv8(x)
        x, self.mem8 = self.lif8(x, self.mem8)
        x = self.conv9(x)
        x, self.mem9 = self.lif9(x, self.mem9)
        x = self.conv10(x) 
        x, self.mem10 = self.lif10(self.mp4(x), self.mem10)
        x = self.conv11(x) 
        x, self.mem11 = self.lif11(x, self.mem11)
        x = self.conv12(x) 
        x, self.mem12 = self.lif12(x, self.mem12)
        x = self.conv13(x) 
        x, self.mem13 = self.lif13(self.mp5(x), self.mem13)

        return x, self.mem13


    def forward_classifier(self, x):
        x = self.flt1(x)
        x = self.lin1(x) 
        # x = self.drp1(x)
        x, self.mem14 = self.lif14(x, self.mem14)
        x = self.lin2(x)
        # x = self.drp2(x) 
        # x, self.mem15 = self.lif15(x, self.mem15)
        # x = self.lin3(x)
        x, self.mem16 = self.lif16(x, self.mem16) 
        return x, self.mem16


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
        # self.mem15 = self.lif15.init_leaky()
        self.mem16 = self.lif16.init_leaky()


    def forward(self, x1):
        spk_rec = []
        self.init_leaky()
        for step in range(x1.size(0)):  # data.size(0) = number of time steps
            x, self.mem13 = self.forward_features(x1[step])
            spk_out, self.mem16 = self.forward_classifier(x)
            spk_rec.append(spk_out)
        return torch.stack(spk_rec)
        
