import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import CharactorVoiceEncoder_MelGAN
from torch.nn.utils import spectral_norm


# for adversarial loss
def adv_loss(logits, target):
    assert target in [1, 0]
    if len(logits.shape) > 1:
        logits = logits.reshape(-1)
    targets = torch.full_like(logits, fill_value=target)
    logits = logits.clamp(min=-10, max=10) # prevent nan
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss

# for R1 regularization loss
def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

# for norm consistency loss
def log_norm(x, mean=-4, std=4, dim=2):
    """
    normalized log mel -> mel -> norm -> log(norm)
    """
    x = torch.log(torch.exp(x * std + mean).norm(dim=dim))
    return x


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == 'none':
            return x
        elif self.layer_type == 'timepreserve':
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == 'half':
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)

class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type

        if self.layer_type == 'none':
            self.conv = nn.Identity()
        elif self.layer_type == 'timepreserve':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 1), stride=(2, 1), groups=dim_in, padding=(1, 0)))
        elif self.layer_type == 'half':
            self.conv = spectral_norm(nn.Conv2d(dim_in, dim_in, kernel_size=(3, 3), stride=(2, 2), groups=dim_in, padding=1))
        else:
            raise RuntimeError('Got unexpected donwsampletype %s, expected is [none, timepreserve, half]' % self.layer_type)
            
    def forward(self, x):
        return self.conv(x)

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample='none'):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance

class Discriminator2d(nn.Module):
    def __init__(self, hps, dim_in=48, num_domains=1, max_conv_dim=384, repeat_num=4):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]

        for lid in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample='half')]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, num_domains, 1, 1, 0))]
        self.main = nn.Sequential(*blocks)
        self.hps = hps
        if hps.character_voice_encoder:
            self.speaker_encoder = CharactorVoiceEncoder_MelGAN(hps)

    def get_feature(self, sample, x):
        features = []
        for i,l in enumerate(self.main):
            if i==2 and self.hps.character_voice_encoder:
                se = self.speaker_encoder(sample)
                x = x + se
            x = l(x)
            features.append(x) 
        out = features[-1]
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        return out, features

    def forward(self, sample,  x):
        out, features = self.get_feature(sample, x)
        out = out.squeeze()  # (batch)
        return out, features

    def get_dloss(self, sample, pd):
        gt = sample['mel'].transpose(1,2)
        pd = pd.transpose(1,2)
        mel_input_length = sample['mel_mask'].sum(1).long()
        mel_len = int(mel_input_length.min().item()) - 1
        pd16k = []
        gt16k = []
        for bib in range(len(mel_input_length)):
            mel_length = mel_input_length[bib]
            random_start = np.random.randint(0, int(mel_length - mel_len))
            pd16k.append(pd[bib, :, random_start:random_start+mel_len])
            gt16k.append(gt[bib, :, random_start:random_start+mel_len])
        pd16k = torch.stack(pd16k)
        gt16k = torch.stack(gt16k).detach()
        gt16k.requires_grad_()
        out, _ = self.forward(sample, gt16k.unsqueeze(1))
        loss_real = adv_loss(out, 1)
        loss_reg = r1_reg(out, gt16k)
        out, _ = self.forward(sample, pd.detach().unsqueeze(1))
        loss_fake = adv_loss(out, 0)
        d_loss = loss_real + loss_fake + loss_reg
        return {'D_loss':d_loss}

    def get_gloss(self, sample, pd):
        gt = sample['mel'].transpose(1,2)
        pd = pd.transpose(1,2)
        mel_input_length = sample['mel_mask'].sum(1).long()
        mel_len = int(mel_input_length.min().item()) - 1
        pd16k = []
        gt16k = []
        for bib in range(len(mel_input_length)):
            mel_length = mel_input_length[bib]
            random_start = np.random.randint(0, int(mel_length - mel_len))
            pd16k.append(pd[bib, :, random_start:random_start+mel_len])
            gt16k.append(gt[bib, :, random_start:random_start+mel_len])
        pd16k = torch.stack(pd16k)
        gt16k = torch.stack(gt16k).detach()
        gt16k.requires_grad_()

        # adversarial loss
        with torch.no_grad():
            _, f_real = self.forward(sample, gt16k.unsqueeze(1))
        out_rec, f_fake = self.forward(sample, pd16k.unsqueeze(1))
        loss_adv = adv_loss(out_rec, 1)
        # feature matching loss
        loss_fm = 0
        for m in range(len(f_real)):
            for k in range(len(f_real[m])):
                loss_fm += torch.mean(torch.abs(f_real[m][k] - f_fake[m][k])) 

        g_loss = 0.2 * loss_adv + 0.2 * 0.1 * loss_fm
        return {'G_loss':g_loss}


