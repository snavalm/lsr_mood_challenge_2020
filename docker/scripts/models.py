
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from collections import OrderedDict

def WNLinear(in_dim, out_dim):
    return nn.utils.weight_norm(nn.Linear(in_dim, out_dim))

class WNConv2d(nn.Module):
    def __init__(self,*args,**kwargs):
        """Weight normalized Conv2d"""
        super().__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(*args,**kwargs))

    def forward(self, input):
        out = self.conv(input)
        return out

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

def down_shift(x):
    x = x[:, :, :-1, :]
    pad = nn.ZeroPad2d((0, 0, 1, 0))
    return pad(x)

def right_shift(x):
    x = x[:, :, :, :-1]
    pad = nn.ZeroPad2d((1, 0, 0, 0))
    return pad(x)

    
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
    
class VQVAE(nn.Module):
    def __init__(self, d, n_channels, code_size, n_block = 3, n_res_block = 4, cond_channels = None, dropout_p = .5,
                 categorical = True, reconstruction_loss = F.cross_entropy):
        super().__init__()
        self.code_size = code_size
        self.d = d
        self.cond_channels = cond_channels
        self.categorical = categorical
        self.reconstruction_loss = reconstruction_loss

        if isinstance(n_channels,int):
            n_channels = [n_channels]*(n_block+1)
        else:
            n_block = len(n_channels)-1

        # Encoder
        down = [nn.Conv2d(d + (0 if cond_channels is None else cond_channels),n_channels[0], kernel_size = 7),
                nn.BatchNorm2d(n_channels[0])]

        for block in range(n_block):
            for res_block in range(n_res_block):
                down.append(GatedResNet(n_channels[block],3,dropout_p=dropout_p,conv=nn.Conv2d, norm= nn.BatchNorm2d))
            down.extend([nn.Conv2d(n_channels[block],n_channels[block+1], kernel_size = 5,stride = 2, padding = 2),
                         nn.BatchNorm2d(n_channels[block+1])])
        down.append(GatedResNet(n_channels[-1],3,dropout_p=dropout_p,conv=nn.Conv2d, norm= nn.BatchNorm2d))

        self.Q = nn.Sequential(*down)

        self.codebook = Quantize(code_size, n_channels[-1])

        # Decoder
        up = [nn.Conv2d(n_channels[-1] + (0 if cond_channels is None else cond_channels),n_channels[-1], kernel_size = 3, padding=1),
                nn.BatchNorm2d(n_channels[-1])]
        for block in range(n_block):
            for res_block in range(n_res_block):
                up.append(GatedResNet(n_channels[-(block+1)],3,dropout_p=dropout_p,conv=nn.Conv2d, norm= nn.BatchNorm2d))
            up.extend([nn.ConvTranspose2d(n_channels[-(block+1)], n_channels[-(block+2)], kernel_size=6, stride=2, padding=2),
                       nn.BatchNorm2d(n_channels[-(block+2)])])
        up.append(GatedResNet(n_channels[0],3,dropout_p=dropout_p,conv=nn.Conv2d, norm= nn.BatchNorm2d))

        up.extend([nn.ELU(),
                   nn.Conv2d(n_channels[0],d,kernel_size=1,padding=0)])
        self.P = nn.Sequential(*up)

    def encode(self, x, cond = None):
        if self.categorical:
            out = F.one_hot(x, self.d).permute(0, 3, 1, 2).contiguous().float()
        else:
            out = x.unsqueeze(1)

        if self.cond_channels is not None:
            cond = cond.float()
            if len(cond.shape) == 2:
                cond = cond.view(cond.shape[0], -1, 1, 1).expand(-1, -1, x.shape[1], x.shape[2])
            out = torch.cat([out, cond], 1)

        return self.Q(out)

    def decode(self, latents, cond = None):
        if self.cond_channels is not None:
            cond = cond.float()
            if len(cond.shape) == 2:
                cond = cond.view(cond.shape[0], -1, 1, 1).expand(-1, -1, latents.shape[2], latents.shape[3])
            latents = torch.cat([latents, cond], 1)

        return self.P(latents)

    def forward(self, x, cond = None):
        z = self.encode(x, cond)
        e, e_st, _ = self.codebook(z)
        x_tilde = self.decode(e_st,cond)

        diff1 = torch.mean((z - e.detach()) ** 2)
        diff2 = torch.mean((e - z.detach()) ** 2)
        return x_tilde, diff1 + diff2

    def loss(self, x, cond = None, reduction = 'mean'):
        x_tilde, diff = self(x, cond)

        if not self.categorical:
            x = x.unsqueeze(1)

        recon_loss = self.reconstruction_loss(x_tilde, x, reduction=reduction)

        if reduction == 'mean':
            loss = recon_loss + diff

        elif reduction == 'none':
            loss = torch.mean(recon_loss) + diff

        elif reduction == 'sum': # This is here for completeness but it doesn't make a lot of sense
            loss = recon_loss/torch.ones_like(recon_loss).sum()  + diff

        return OrderedDict(loss=loss, recon_loss=recon_loss, reg_loss=diff)

def causal_mask(size):
    shape = [size, size]
    mask = np.triu(np.ones(shape), k=1).astype(np.uint8).T
    start_mask = np.ones(size).astype(np.float32)
    start_mask[0] = 0

    return (
        torch.from_numpy(mask).unsqueeze(0),
        torch.from_numpy(start_mask).unsqueeze(1),
    )

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

        #print(filter[0,0])
        x = F.conv2d(input=x, weight=filter,
                     bias=self.bias,
                     stride=self.stride,
                     padding= (self.kernel_size[0]//2,self.kernel_size[1]//2)) # Same padding
        return x

    
class PixelSNAIL(nn.Module):
    def __init__(self,d,
        shape = (64,64),n_channels=64,n_block=4,n_res_block = 2, ini_kernel_size = 7,dropout_p=0.1,
        cond_channels=None, downsample = 1, downsample_attn = 2, non_linearity = F.elu):
        super().__init__()

        self.non_linearity = non_linearity
        height, width = shape

        self.d = d
        self.cond_channels = cond_channels
        self.ini_conv = MaskedConv(d,n_channels, kernel_size=7, stride = downsample, mask_type='A',)
        self.ini_conv = MaskedConv(d, n_channels, kernel_size=ini_kernel_size, stride = downsample, mask_type='A',)

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
            self.blocks.append(PixelBlock(n_channels,n_channels, n_res_block=n_res_block, shape = (height,width),
                                          downsample_attn = downsample_attn, dropout_p=dropout_p,
                                          cond_channels=cond_channels,non_linearity = non_linearity))

        self.upsample = nn.ConvTranspose2d(n_channels, n_channels,kernel_size=downsample, stride=downsample)

        self.out = WNConv2d(n_channels, d, 1)

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
    
class VQLatentSNAIL(PixelSNAIL):
    def __init__(self, feature_extractor_model, **kwargs):
        super().__init__(d = feature_extractor_model.code_size,
                                       **kwargs)

        for p in feature_extractor_model.parameters():
            p.requires_grad = False

        self.feature_extractor_model = feature_extractor_model
        self.feature_extractor_model.eval()

    def retrieve_codes(self,x,cond):
        with torch.no_grad():
            self.feature_extractor_model.eval()
            z = self.feature_extractor_model.encode(x,cond)
            _,_,code = self.feature_extractor_model.codebook(z)
        return code

    def forward(self, x, cond = None):
        # Retrieve codes for images
        code = self.retrieve_codes(x,cond)
        return super(VQLatentSNAIL,self).forward(code,cond)

    def forward_latent(self, code, cond = None):
        return super(VQLatentSNAIL,self).forward(code,cond)

    def loss(self, x, cond = None, reduction = 'mean'):
        # Retrieve codes for images
        code = self.retrieve_codes(x,cond)
        logits = super(VQLatentSNAIL,self).forward(code, cond)
        nll = F.cross_entropy(logits, code, reduction = reduction)
        return OrderedDict(loss=nll)

    def sample(self, n, img_size = (64,64), cond = None):
        device = next(self.parameters()).device
        samples = torch.zeros(n, *img_size).long().to(device)
        with torch.no_grad():
            for r in range(img_size[0]):
                for c in range(img_size[1]):
                    if self.cond_channels is not None:
                        logits = super(VQLatentSNAIL,self).forward(samples,cond)[:, :, r, c]
                    else:
                        logits = super(VQLatentSNAIL,self).forward(samples)[:, :, r, c]
                    probs = F.softmax(logits, dim=1)
                    samples[:, r, c] = torch.multinomial(probs, 1).squeeze(-1)
        return samples.cpu().numpy()
    