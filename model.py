import torch
import torch.nn.functional as F
import torch.nn as nn
import math
from utils import capture_init
from module import upsample2, P_Conv, N_Conv, PNP_Conv_operation, rescale_module,\
                    LSTM, SeparableConvBlock, MemoryEfficientSwish, Swish
import pdb


class Analysis_stage(nn.Module):
    @capture_init
    def __init__(self, chin=1, hidden=6, kernel_size=4, stride=4, resample=4, depth=5,
                normalize=True, rescale=0.1, floor=1e-3, pfactor = [17, 13, 11, 7, 5], npfactor = [0.2]):
        super().__init__()
        self.chin = chin
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.stride = stride
        self.resample = 4
        self.depth = 5
        self.normalize = normalize
        self.floor = floor
        self.pfactor = pfactor
        self.npfactor = npfactor
        self.p_conv1 = nn.ModuleList()
        self.np_conv1 = nn.ModuleList()
        self.p_conv2 = nn.ModuleList()
        self.np_conv2 = nn.ModuleList()
        self.p_convs = nn.ModuleList()
        
        # PNP-Conv1
        pconv1 = P_Conv(self.chin, self.hidden, self.kernel_size, self.stride, self.pfactor[0])
        npconv1 = N_Conv(self.chin, self.hidden, self.kernel_size, self.stride, self.npfactor[0])
        
        # PNP-Conv2
        pconv2 = P_Conv(self.hidden, self.hidden*2, self.kernel_size, self.stride, self.pfactor[1])
        npconv2 = N_Conv(self.hidden, self.hidden*2, self.kernel_size, self.stride, self.npfactor[0])
        
        # P-Conv3, 4, 5
        pconv3 = P_Conv(self.hidden*2, self.hidden*4, self.kernel_size*2, self.stride, self.pfactor[1])
        pconv4 = P_Conv(self.hidden*4, self.hidden*8, self.kernel_size*2, self.stride, self.pfactor[1])
        pconv5 = P_Conv(self.hidden*8, self.hidden*16, self.kernel_size*3, self.stride, self.pfactor[1])

        self.p_conv1.append(nn.Sequential(*pconv1))
        self.np_conv1.append(nn.Sequential(*npconv1))
        self.p_conv2.append(nn.Sequential(*pconv2))
        self.np_conv2.append(nn.Sequential(*npconv2))
        self.p_convs.append(nn.Sequential(*pconv3))
        self.p_convs.append(nn.Sequential(*pconv4))
        self.p_convs.append(nn.Sequential(*pconv5))
        self.lstm = LSTM(hidden*16, bi=False)
        
        if rescale:
            rescale_module(self, reference=rescale)

    def valid_length(self, length):
        length = math.ceil(length * self.resample)
        for idx in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(length, 1)
        for idx in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size
        length = int(math.ceil(length / self.resample))
        return int(length)

    def forward(self, wav, condition=torch.tensor([1, 1])):
        if wav.dim() == 2:
            wav = wav.unsqueeze(1)
        if self.normalize:
            mono = wav.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            wav = wav / (self.floor + std)
        else:
            std = 1
            
        length = wav.shape[-1]
        x = wav
        x = F.pad(x, (0, self.valid_length(length) - length))
        
        x = upsample2(x)
        x = upsample2(x)

        multi_feat = []
        
        x = PNP_Conv_operation(x, self.p_conv1, self.np_conv1)
        x = PNP_Conv_operation(x, self.p_conv2, self.np_conv2)
        multi_feat.append(x)
        
        for p in self.p_convs:
            x = p(x)
            multi_feat.append(x)
            
        x = x. permute(2,0,1)
        x, _ = self.lstm(x)
        x = x.permute(1,2,0)
        multi_feat.append(x)
        
        return multi_feat


class Estimation_stage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 48
        self.a_stage = Analysis_stage()
        self.light_bifpn = Light_BiFPN(num_channels=num_channels)
        self.ln = (nn.Linear(in_features=240, out_features=360))

    def forward(self, input):        
        p1, p2, p3, p4, p5 = self.a_stage(input)
        # [B,12,16020], [B,24,4004], [B,48,1000], [B,96,249], [B,96,249]
        
        p1 = p1[:,:,:,None]
        p2 = p2[:,:,:,None]
        p3 = p3[:,:,:,None]
        p4 = p4[:,:,:,None]
        p5 = p5[:,:,:,None]

        features = (p1, p2, p3, p4, p5)
        f0_features = self.light_bifpn(features)
        # [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1]
        " 500 = frame number while the input_length is 4sec & hop_lenghth is 128 & sampling rate is 16 kHz"

        f0_feature = torch.cat(f0_features, dim=1)
        # f0_feature: [B, 48*5=240, 500, 1]
        f0_feature = f0_feature.permute(0,2,1,3).squeeze(3)
        # f0_feature: [B, 500, 240]
        f0out = self.ln(f0_feature)
        # [B, 500, 360]
        f0out = torch.sigmoid(f0out)

        return f0out


class Light_BiFPN(nn.Module):
    def __init__(self, num_channels, epsilon=1e-8, orig_swish=False):
        super(Light_BiFPN, self).__init__()

        self.epsilon = epsilon
        
        # Pre resize
        self.p1_upch = SeparableConvBlock(num_channels//4, num_channels)
        self.p2_upch = SeparableConvBlock(num_channels//2, num_channels)
        self.p4_dnch = SeparableConvBlock(num_channels*2, num_channels)
        self.p5_dnch = SeparableConvBlock(num_channels*2, num_channels)

        self.p4_upsample = nn.Upsample(scale_factor=(2,1), mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=(2,1), mode='nearest')

        # BiFPN conv layers
        self.conv6_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv3_up = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv4_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv5_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv6_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
        self.conv7_down = SeparableConvBlock(num_channels, orig_swish=orig_swish)
    

        self.swish = MemoryEfficientSwish() if not orig_swish else Swish()

        # Weight
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p2_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p2_w1_relu = nn.ReLU()
        self.p1_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p1_w1_relu = nn.ReLU()

        self.p2_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p2_w2_relu = nn.ReLU()
        self.p3_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p3_w2_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

    def pre_resize(self, inputs):
        # [B,12,16020], [B,24,4004], [B,48,1000], [B,96,249], [B,96,249]
        p1, p2, p3, p4, p5 = inputs
        
        # Pre resizing
        p1 = self.p1_upch(F.avg_pool2d(p1, (32,1)))
        # f1: [B,48,500,1]
        p2 = F.pad(self.p2_upch(F.avg_pool2d(p2, (8,1))), (0,0,1,0))
        # f2: [B,48,500,1]
        p3 = F.pad(F.avg_pool2d(p3, (2,1)), (0,0,1,0))
        # f3: [B,48,500,1]
        p4 = self.p4_dnch(self.p4_upsample(F.pad(p4,(0,0,2,1))))
        # f4: [B,48,500,1]
        p5 = self.p5_dnch(self.p5_upsample(F.pad(p5,(0,0,2,1))))
        # f5: [B,48,500,1]
        return p1, p2, p3, p4, p5
        

    def forward(self, inputs):
        # The BiFPN illustration is an upside down form of the figure in the paper.
        """
        Illustration of a bifpn layer unit
            p5_in ---------------------------> p5_out -------->
               |---------------|                  ↑
                               ↓                  |
            p4_in ---------> p4_mid ---------> p4_out -------->
               |---------------|----------------↑ ↑
                               ↓                  |
            p3_in ---------> p3_mid ---------> p3_out -------->
               |---------------|----------------↑ ↑
                               ↓                  |
            p2_in ---------> p2_mid ---------> p2_out -------->
               |---------------|----------------↑ ↑
                               |------------ ---↓ |
            p1_in ---------------------------> p1_out -------->
        """
        
        p1_in, p2_in, p3_in, p4_in, p5_in = self.pre_resize(inputs)
        # [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1], [B,48,500,1]
        
        # BiFPN operation
        ## Top-bottom process
        # Weights for p4_in and p5_in to p4_mid
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        # Connections for p4_in and p5_in to p4_mid respectively
        p4_mid = self.conv6_up(self.swish(weight[0] * p4_in + weight[1] * (p5_in)))

        # Weights for p3_in and p4_mid to p3_mid
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        # Connections for p3_in and p4_mid to p3_mid respectively
        p3_mid = self.conv5_up(self.swish(weight[0] * p3_in + weight[1] * (p4_mid)))

        # Weights for p2_in and p3_mid to p2_mid
        p2_w1 = self.p2_w1_relu(self.p2_w1)
        weight = p2_w1 / (torch.sum(p2_w1, dim=0) + self.epsilon)
        # Connections for p2_in and p3_mid to p2_mid respectively
        p2_mid = self.conv4_up(self.swish(weight[0] * p2_in + weight[1] * (p3_mid)))

        # Weights for p1_in and p2_mid to p1_out
        p1_w1 = self.p1_w1_relu(self.p1_w1)
        weight = p1_w1 / (torch.sum(p1_w1, dim=0) + self.epsilon)
        # Connections for p1_in and p2_mid to p1_out respectively
        p1_out = self.conv3_up(self.swish(weight[0] * p1_in + weight[1] * (p2_mid)))

        ## Down-Up process
        # Weights for p2_in, p2_mid and p1_out to p2_out
        p2_w2 = self.p2_w2_relu(self.p2_w2)
        weight = p2_w2 / (torch.sum(p2_w2, dim=0) + self.epsilon)
        # Connections for p2_in, p2_mid and p1_out to p2_out respectively
        p2_out = self.conv4_down(self.swish(weight[0] * p2_in + weight[1] * p2_mid + weight[2] * (p1_out)))

        # Weights for p3_in, p3_mid and p2_out to p3_out
        p3_w2 = self.p3_w2_relu(self.p3_w2)
        weight = p3_w2 / (torch.sum(p3_w2, dim=0) + self.epsilon)
        # Connections for p3_in, p3_mid and p2_out to p3_out respectively
        p3_out = self.conv5_down(self.swish(weight[0] * p3_in + weight[1] * p3_mid + weight[2] * (p2_out)))

        # Weights for p4_in, p4_mid and p3_out to p4_out
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        # Connections for p4_in, p4_mid and p3_out to p4_out respectively
        p4_out = self.conv6_down(self.swish(weight[0] * p4_in + weight[1] * p4_mid + weight[2] * (p3_out)))

        # Weights for p5_in and p4_out to p5_out
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        # Connections for p5_in and p4_out to p5_out
        p5_out = self.conv7_down(self.swish(weight[0] * p5_in + weight[1] * (p4_out)))

        return p1_out, p2_out, p3_out, p4_out, p5_out


