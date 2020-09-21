from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import nn_blocks

class PixelSNAIL(nn.Module):
    def __init__(self,d,
        shape = (64,64),n_channels=64, n_block=4,n_res_block = 2, dropout_p=0.1,
        cond_channels=None, downsample = 1, non_linearity = F.elu):
        super().__init__()

        self.non_linearity = non_linearity
        height, width = shape

        self.d = d
        self.cond_channels = cond_channels
        self.ini_conv = nn_blocks.MaskedConv(d,n_channels, kernel_size=7, stride = downsample, mask_type='A',)

        height //= downsample
        width //= downsample

        # Creates a grid with coordinates within image
        coord_x = (torch.arange(height).float() - height / 2) / height
        coord_x = coord_x.view(1, 1, height, 1).expand(1, 1, height, width)
        coord_y = (torch.arange(width).float() - width / 2) / width
        coord_y = coord_y.view(1, 1, 1, width).expand(1, 1, height, width)
        self.register_buffer('background', torch.cat([coord_x, coord_y], 1))

        self.blocks = nn.ModuleList()

        for i in range(n_block):
            self.blocks.append(nn_blocks.PixelBlock(n_channels,n_channels, n_res_block=n_res_block, shape = (height,width),
                                          dropout_p=dropout_p,cond_channels=cond_channels,non_linearity = non_linearity))

        self.upsample = nn.ConvTranspose2d(n_channels, n_channels,kernel_size=downsample, stride=downsample)

        self.out = nn_blocks.WNConv2d(n_channels, d, 1)

    def forward(self, input, cond=None):

        input = F.one_hot(input, self.d).permute(0, 3, 1, 2).type_as(self.background)

        if self.cond_channels is not None:
            cond = cond.float()

        out = self.ini_conv(input)

        batch, _, height, width = out.shape
        background = self.background.expand(batch, -1, -1, -1)

        for block in self.blocks:
            out = block(out, background=background, cond=cond)

        out = self.upsample(self.non_linearity(out))
        out = self.out(self.non_linearity(out))

        return out

    def loss(self, x, cond = None, reduction = 'mean'):
        logits = self.forward(x, cond)
        nll = F.cross_entropy(logits, x,reduction=reduction)
        return OrderedDict(loss=nll)

    def sample(self, n, img_size = (64,64), cond = None):
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)
        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    if self.cond_channels is not None:
                        logits = self(samples,cond)[:, :, r, c]
                    else:
                        logits = self(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.cpu().numpy()
