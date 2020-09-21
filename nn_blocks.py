import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class WNConv2d(nn.Module):
    def __init__(self,*args,**kwargs):
        """Weight normalized Conv2d"""
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(*args,**kwargs))

    def forward(self, input):
        out = self.conv(input)
        return out

def WNLinear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))

class nin(nn.Module):
    def __init__(self,in_channels,out_channels):
        """Network in Network, 1x1 Convolution """
        super().__init__()
        self.out_channels = out_channels
        self.linear = WNLinear(in_channels,out_channels)
    def forward(self,x):
        batch, c, height, width = x.shape
        x = x.view(batch,c,-1).transpose(1, 2)
        x = self.linear(x)
        return x.view(batch,self.out_channels,height,width)

class MaskedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, mask_type, stride = 1, v_gated = False, cond_channels = None):
        super(MaskedConv, self).__init__()

        if isinstance(kernel_size,int):
            kernel_size = (kernel_size,kernel_size)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.stride = stride

        if v_gated:
            self.weight_size = kernel_size[0] * (kernel_size[1] // 2)
        else:
            self.weight_size = kernel_size[0] * (kernel_size[1] // 2) + kernel_size[0] // 2
            if mask_type == 'B':
                self.weight_size += 1

        self.bias = nn.Parameter(torch.zeros((out_channels)), requires_grad=True)
        self.weight = nn.Parameter(torch.randn((out_channels, in_channels, self.weight_size)) * 0.01,
                                   requires_grad=True)
        self.register_buffer('rest_of_filter',torch.zeros((out_channels, in_channels,
                                                           kernel_size[0]*kernel_size[1] - self.weight_size)))

        if cond_channels is not None:
            self.weight_cond = nn.Parameter(torch.randn((out_channels, cond_channels,
                                                         kernel_size[0], kernel_size[1])) * 0.01,
                                            requires_grad=True)

    def forward(self, x, cond = None):
        filter = torch.cat([self.weight,self.rest_of_filter],dim=2)
        filter = filter.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        if self.cond_channels is not None:
            filter = torch.cat([filter,self.weight_cond],dim=1)

            # If condition is flat reshape and expand so it can be concatenated
            if (len(cond.shape)==2):
                cond = cond.view(cond.shape[0],-1,1,1).expand(-1,-1,x.shape[2],x.shape[3])
            # Append condition to input
            x = torch.cat([x,cond],1)

        x = F.conv2d(input=x, weight=filter,
                     bias=self.bias,
                     stride=self.stride,
                     padding= (self.kernel_size[0]//2,self.kernel_size[1]//2)) # Same padding
        return x

class ResMaskedBlock(nn.Module):
    def __init__(self, in_channels, res_channels, kernel_size, cond_channels = None, non_linearity = F.relu):
        super(ResMaskedBlock, self).__init__()
        self.cond_channels = cond_channels
        self.non_linearity = non_linearity

        self.iniconv = MaskedConv(in_channels=in_channels,out_channels=res_channels,kernel_size=1,mask_type='B',
                                  cond_channels=cond_channels)
        self.midconv = MaskedConv(in_channels=res_channels,out_channels=res_channels,kernel_size=kernel_size,mask_type='B')
        self.endconv = MaskedConv(in_channels=res_channels, out_channels=in_channels,kernel_size= 1, mask_type='B')

    def forward(self,x, cond = None):
        out = x
        out = self.non_linearity(self.iniconv(out,cond))
        out = self.non_linearity(self.midconv(out))
        out = self.non_linearity(self.endconv(out))
        out += x
        return out

def down_shift(x):
    x = x[:, :, :-1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0))
    return pad(x)

def right_shift(x):
    x = x[:, :, :, :-1]
    pad = nn.ZeroPad2d((1, 0, 0, 0))
    return pad(x)

class GatedMaskedConv(nn.Module):
    def __init__(self, in_channels, kernel_size,cond_channels = None):
        """Implementation of gated residual unit of https://arxiv.org/pdf/1606.05328.pdf"""
        super(GatedMaskedConv, self).__init__()

        self.cond_channels = cond_channels

        # Initial layer
        self.v_conv = MaskedConv(in_channels,in_channels*2,kernel_size,'B',v_gated=True,cond_channels=cond_channels)
        self.h_conv = MaskedConv(in_channels, in_channels*2,(1,kernel_size),'B',cond_channels=cond_channels)
        self.int_conv = nn.Conv2d(in_channels*2,in_channels*2,1)
        self.out_conv = nn.Conv2d(in_channels,in_channels,1)

    def forward(self,x, cond = None):
        xv,xh = x.chunk(2,dim=1)

        # Initial Conv Vertical stack
        xv = self.v_conv(xv,cond)

        # Horizontal stack
        xh_conv = self.h_conv(xh,cond)
        xh_conv += self.int_conv(down_shift(xv))
        xh_conv_tanh, xv_conv_sigmoid = xh_conv.chunk(2,dim=1)
        xh_conv = torch.tanh(xh_conv_tanh) * torch.sigmoid(xv_conv_sigmoid)
        xh_conv = self.out_conv(xh_conv)
        xh += xh_conv

        # Output of Vertical stack
        xv_tanh,xv_sigmoid = xv.chunk(2,dim=1)
        xv = torch.tanh(xv_tanh) * torch.sigmoid(xv_sigmoid)

        return torch.cat((xv,xh),dim=1)


class GatedResNet(nn.Module):
    def __init__(self, in_channels, kernel_size, n_channels = None, aux_channels = None,
                 cond_channels = None, dropout_p = 0.,non_linearity = F.elu, conv = WNConv2d, norm = None):

        """Implementation of gated residual unit of https://openreview.net/pdf?id=BJrFC6ceg
        Note that this is not masked, for being causal it relies on specific conv sizes and shifts outside of the
        scope of this module """
        super(GatedResNet, self).__init__()

        if n_channels is None:
            n_channels = in_channels

        self.n_channels = n_channels
        self.aux_channels = aux_channels
        self.cond_channels = cond_channels
        self.dropout_p = dropout_p
        self.non_linearity = non_linearity

        self.kernel_size = (kernel_size,kernel_size) if isinstance(kernel_size,int) else kernel_size
        self.pad = [i//2 for i in self.kernel_size]

        self.conv1 = conv(in_channels + (0 if cond_channels is None else cond_channels),
                               n_channels, kernel_size, padding=self.pad)
        self.norm1 = None if norm is None else norm(n_channels)

        if aux_channels is not None:
            self.conv1_aux = conv(aux_channels, n_channels, 1)

        self.conv2 = conv(n_channels,in_channels * 2, kernel_size, padding=self.pad)
        self.norm2 = None if norm is None else norm(in_channels * 2)

        if dropout_p > 0.:
            self.dropout = nn.Dropout2d(dropout_p)

        self.gate = nn.GLU(1)

    def forward(self, x, aux = None, cond = None):
        out = x
        # If there are conditional channels, append them
        # NOTE: In the original paper they add conditions after conv2.
        # They also pass it through a 1f conv before adding it
        if self.cond_channels is not None:
            if len(cond.shape) == 2:
                cond = cond.view(cond.shape[0], -1, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
            out = torch.cat([out, cond], 1)

        if self.norm1 is not None:
            out = self.norm1(out)
        out = self.conv1(self.non_linearity(out))

        # Filters of size 2 increase space by 1 h, get rid of the additional
        if self.kernel_size[0] == 2:
            out = out[:,:,:-1]
        if self.kernel_size[1] == 2:
            out = out[:, :, :,:-1]
        # Add auxiliary channels
        if self.aux_channels is not None:
            out += self.conv1_aux(self.non_linearity(aux))
        out = self.non_linearity(out)

        if self.dropout_p>0:
            out = self.dropout(out)

        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        # If kernel size is pair, padding above adds one dimension. This gets rid of it
        if self.kernel_size[0] == 2:
            out = out[:,:,:-1]
        if self.kernel_size[1] == 2:
            out = out[:, :, :,:-1]

        out = self.gate(out)
        return out + x


# FROM https://github.com/rosinality/vq-vae-2-pytorch/blob/master/pixelsnail.py
def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),
        torch.from_numpy(start_mask).unsqueeze(1),
    )

#%%

class CausalAttention(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, key_size = 16,
                 value_size = 128, n_head=2, shape = (16,16)):
        super().__init__()
        self.value_size = value_size
        self.key_size = key_size
        self.n_head = n_head
        self.dim_head = key_size * n_head

        query = [GatedResNet(query_in_channels, kernel_size=1,n_channels=query_in_channels),
                 nin(query_in_channels, self.dim_head)]
        self.query = nn.Sequential(*query)

        key = [GatedResNet(key_in_channels, kernel_size=1, n_channels=key_in_channels),
               nin(key_in_channels, self.dim_head)]
        self.key = nn.Sequential(*key)

        mask, start_mask = causal_mask(shape[0] * shape[1])
        self.register_buffer('mask', mask)
        self.register_buffer('start_mask', start_mask)

        value = [GatedResNet(key_in_channels, kernel_size=1, n_channels=key_in_channels),
               nin(key_in_channels, value_size * n_head)]
        self.value = nn.Sequential(*value)

    def forward(self, query, key):
        batch, _, height, width = key.shape

        query = self.query(query)
        value = self.value(key)
        key = self.key(key)

        # Reshape and transpose as needed
        query = query.view(batch, -1, self.n_head, self.key_size).transpose(1, 2)
        key = key.view(batch, -1, self.n_head, self.key_size).permute(0,2,3,1)
        value = value.view(batch, -1, self.n_head, self.value_size).transpose(1, 2)

        attn = torch.matmul(query, key) / np.sqrt(self.dim_head)

        mask, start_mask = self.mask, self.start_mask
        mask = mask.type_as(query)
        start_mask = start_mask.type_as(query)

        attn = attn.masked_fill(mask == 0, -1e4)
        attn = torch.softmax(attn, 3) * start_mask

        out = attn @ value
        out = out.transpose(1, 2).reshape(batch, height, width, self.value_size * self.n_head)
        out = out.permute(0, 3, 1, 2)

        return out

class PixelBlock(nn.Module):
    def __init__(self,in_channel,value_size = 80, n_res_block=4, shape=(32,32), dropout_p=0.1,cond_channels=None,
                 non_linearity = F.elu, downsample_attn = 2):
        super().__init__()

        self.non_linearity = non_linearity

        v_resblocks = []
        h_resblocks = []
        for _ in range(n_res_block):
            v_resblocks.append(GatedResNet(in_channels = in_channel, kernel_size = [2,3],
                                           n_channels = in_channel, cond_channels=cond_channels,
                                           dropout_p=dropout_p, non_linearity=non_linearity))
            h_resblocks.append(GatedResNet(in_channels = in_channel, kernel_size = [2,2],
                                           n_channels= in_channel, cond_channels=cond_channels,
                                           aux_channels= in_channel,
                                           dropout_p = dropout_p, non_linearity=non_linearity))

        self.v_resblocks = nn.ModuleList(v_resblocks)
        self.h_resblocks = nn.ModuleList(h_resblocks)

        self.downsample_key = MaskedConv(in_channel * 2 + 2,in_channel, kernel_size=5,
                                         stride = downsample_attn, mask_type='B',)
        self.downsample_query = MaskedConv(in_channel + 2,in_channel, kernel_size=5,
                                           stride = downsample_attn, mask_type='B',)

        shape_attn = (shape[0]//downsample_attn,shape[1]//downsample_attn)
        self.causal_attention = CausalAttention(key_in_channels = in_channel,
                                                query_in_channels = in_channel,
                                                value_size=value_size,
                                                n_head=1, shape = shape_attn)

        self.upsample = nn.ConvTranspose2d(value_size, in_channel,kernel_size=downsample_attn,stride=downsample_attn)

        self.out_resblock = GatedResNet(in_channel, kernel_size=1, n_channels=in_channel,
                                        aux_channels= in_channel,dropout_p=dropout_p)

    def forward(self, input, background, cond=None):
        out = input
        v_out = h_out = out
        for v_resblock, h_resblock in zip(self.v_resblocks,self.h_resblocks):
            v_out = v_resblock(v_out, cond=cond)
            h_out = h_resblock(h_out, aux = down_shift(v_out), cond=cond)

        out = h_out

        # Get rid of intermediate variables to free memroy
        del h_out
        del v_out

        key = self.non_linearity(self.downsample_key(torch.cat([input, out, background], 1)))
        query = self.non_linearity(self.downsample_query(torch.cat([out, background], 1)))
        attn_out = self.causal_attention(query, key)
        attn_out = self.upsample(self.non_linearity(attn_out))

        out = self.out_resblock(out, attn_out)

        return out


class Quantize(nn.Module):

    def __init__(self, size, code_dim):
        super().__init__()
        self.embedding = nn.Embedding(size, code_dim)
        self.embedding.weight.data.uniform_(-1./size,1./size)

        self.code_dim = code_dim
        self.size = size

    def forward(self, z):
        b, c, h, w = z.shape
        weight = self.embedding.weight

        flat_inputs = z.permute(0, 2, 3, 1).contiguous().view(-1, self.code_dim)
        distances = (flat_inputs ** 2).sum(dim=1, keepdim=True) \
                    - 2 * torch.mm(flat_inputs, weight.t()) \
                    + (weight.t() ** 2).sum(dim=0, keepdim=True)
        encoding_indices = torch.max(-distances, dim=1)[1]
        encoding_indices = encoding_indices.view(b, h, w)
        quantized = self.embedding(encoding_indices).permute(0, 3, 1, 2).contiguous()

        return quantized, (quantized - z).detach() + z, encoding_indices
