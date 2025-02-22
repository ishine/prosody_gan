import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class Dblock(nn.Module):
    def __init__(self, n, c_dim, ks=3):
        super().__init__()
        self.conv_ss1 = weight_norm(nn.Conv1d(n, 2 * n, kernel_size=ks, padding=(ks - 1) // 2))
        self.conv_ss2 = weight_norm(nn.Conv1d(2 * n, 2 * n, kernel_size=ks, padding=(ks - 1) // 2))
        self.conv_cs1 = weight_norm(nn.Conv1d(c_dim, 2 * n, kernel_size=ks, padding=(ks - 1) // 2))
        self.conv_cs2 = weight_norm(nn.Conv1d(2 * n, c_dim, kernel_size=ks, padding=(ks - 1) // 2))

    def forward(self, x, c):
        xx = self.conv_ss1(x)
        cc = self.conv_cs1(c)
        xx = self.conv_ss2(F.leaky_relu(xx + cc,  0.1)) + xx
        cc = self.conv_cs2(F.leaky_relu(cc,  0.1)) + c
        return xx, cc


class MultiSpectroDiscriminator(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.blocks = nn.ModuleList()
        for i in range(4):
            self.blocks.append(Dblock(4 * (2**i), 256, ks=3))
        self.conv_post = weight_norm(nn.Conv1d(4 * 2 * (2**i), 1, kernel_size=1))

    def forward(self, x, c, mask):
        fmap = []
        for l in self.blocks:
            x, c = l(x, c)
            x = x * mask
            c = c * mask
            fmap.append(x)
        x = self.conv_post(x)
        return x, fmap


class MultiSpectroGAN(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.discriminators = nn.ModuleList()
        self.d = 2
        for _ in range(1):
            self.discriminators.append(MultiSpectroDiscriminator(hparams))

    def downsample(self, x,bool=False):
        #(b,t,c)
        if len(x.shape) == 3:
            b,c,t = x.shape
            if t % self.d != 0:
                zeros = torch.zeros([b ,c,self.d - t % self.d]).to(x)
                x = torch.cat([x,zeros],-1)
            if bool:
                x = x.reshape(b,c,-1,self.d).all(-1)
            else:
                x = x.reshape(b,c,-1,self.d).mean(-1)
        return x

    def forward(self, y, y_hat, c, mask, d_loss):
        y, y_hat, c = y.transpose(1,2), y_hat.transpose(1,2), c.transpose(1,2)
        mask = mask.unsqueeze(1)
        watches = {}
        if d_loss:
            watches['D_loss'] = 0
            for i, d in enumerate(self.discriminators):
                if i>=1:
                    y_hat = self.downsample(y_hat)
                    y = self.downsample(y)
                    c = self.downsample(c)
                    mask = self.downsample(mask,True)
                y_d_r, _ = d(y, c, mask)
                y_d_g, _ = d(y_hat.detach(), c, mask)
                watches['D_loss%d_r'%i], watches['D_loss%d_g'%i] = discriminator_loss(y_d_r, y_d_g, mask)
                watches['D_loss'] += watches['D_loss%d_r'%i] + watches['D_loss%d_g'%i]
            return watches
        else:
            watches['G_loss'] = 0
            watches['FM_loss'] = 0
            for i, d in enumerate(self.discriminators):
                if i>=1:
                    y_hat = self.downsample(y_hat)
                    y = self.downsample(y)
                    c = self.downsample(c)
                    mask = self.downsample(mask,True)
                y_d_r, fmap_r = d(y, c, mask)
                y_d_g, fmap_g = d(y_hat, c, mask)
                watches['FM_loss%d'%i] = feature_loss(fmap_r, fmap_g, mask)
                watches['G_loss%d'%i] = generator_loss(y_d_g, mask)
                watches['FM_loss'] += watches['FM_loss%d'%i]
                watches['G_loss'] += watches['G_loss%d'%i]
            return watches

        
        
def discriminator_loss(dr, dg, mask):
    r_loss = torch.sum((1 - dr)**2 * mask) / mask.sum()
    g_loss = torch.sum(dg**2 * mask) / mask.sum()
    return r_loss, g_loss


def feature_loss(fmap_r, fmap_g, mask):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        loss += torch.sum(torch.abs(dr - dg)*mask) / mask.sum()
    return loss

def generator_loss(dg, mask):
    return torch.sum((1 - dg)**2* mask) / mask.sum()
