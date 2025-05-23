import torch
import torch.nn as nn
from droid_slam.modules.kan import KANLinear

class KAN_bias_GRU(nn.Module):
    def __init__(self, h_planes=128, i_planes=128):
        super(KAN_bias_GRU, self).__init__()
        self.do_checkpoint = False

        self.convz = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convr = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)
        self.convq = nn.Conv2d(h_planes+i_planes, h_planes, 3, padding=1)

        self.kanz_glo = KANLinear(128, 128,grid_size=3)
        self.kanr_glo = KANLinear(128, 128,grid_size=3)
        self.kanq_glo = KANLinear(128, 128,grid_size=3)

        self.w = nn.Conv2d(h_planes, h_planes, 1, padding=0)

    def forward(self, net, *inputs):
        inp = torch.cat(inputs, dim=1)
        net_inp = torch.cat([net, inp], dim=1)

        b, c, h, w = net.shape
        glo = torch.sigmoid(self.w(net)) * net
        glo = glo.view(b, c, h*w).mean(-1).view(b, c)

        kanz_glo = self.kanz_glo(glo).view(b, c,1,1)
        kanr_glo = self.kanr_glo(glo).view(b, c,1,1)
        kanq_glo = self.kanq_glo(glo).view(b, c,1,1)

        z = torch.sigmoid(self.convz(net_inp) + kanz_glo)
        r = torch.sigmoid(self.convr(net_inp) + kanr_glo)
        q = torch.tanh(self.convq(torch.cat([r*net, inp], dim=1)) + kanq_glo)

        net = (1-z) * net + z * q #
        return net
