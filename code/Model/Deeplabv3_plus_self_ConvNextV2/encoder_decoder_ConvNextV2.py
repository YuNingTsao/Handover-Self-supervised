import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
# from config import config
from Utils.self_mask import *


from MinkowskiEngine import (
    MinkowskiConvolution,
    MinkowskiDepthwiseConvolution,
    MinkowskiLinear,
)

from timm.models.layers import trunc_normal_
from .convnextv2_sparse import SparseConvNeXtV2
from .convnextv2 import Block
import timm

bn_eps = 1e-5
bn_momentum = 0.1

backbone = tiim.create_model('convnextv2_base', pretrained=True, features_only=True, out_indices=(0,1,2,3,4))

# Encoder
class EncoderNetwork(nn.Module):
    def __init__(self, num_classes):
        super(EncoderNetwork, self).__init__()
        self.backbone = backbone

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 2
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask
    
    def forward(self, data):
        patch_data = self.patchify(data)
        masked_data = self.gen_random_mask(patch_data)
        x, mask = self.backbone(masked_data)
        return x, mask




#Decoder
class DecoderNetwork(nn.Module):
    def __init__(self, decoder_depth=1,
                decoder_embed_dim=512):
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth

           # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    

    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    


    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, x, mask): 
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(x, pred, mask)
        return loss, pred, mask

#VAT
def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def get_r_adv_t(x, x1, decoder1, decoder2, it=1, xi=1e-1, eps=10.0):
    # stop bn
    decoder1.eval()
    decoder2.eval()
    
    x_detached = x.detach()
    x1_detached = x1.detach()
    with torch.no_grad():
        # get the ensemble results from teacher
        _, de_x = decoder1(x_detached)
        _, de_x1 = decoder2(x1_detached)
        pred = F.softmax(( de_x + de_x1 )/2, dim=1)

    d = torch.rand(x.shape).sub(0.5).to(x.device)
    d = _l2_normalize(d)

    # assist students to find the effective va-noise
    for _ in range(it):
        d.requires_grad_()
        _, de_x = decoder1(x_detached + xi * d)
        _, de_x1 = decoder2(x1_detached + xi * d)
        pred_hat = ( de_x + de_x1 )/2
        logp_hat = F.log_softmax(pred_hat, dim=1)
        adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
        adv_distance.backward()
        d = _l2_normalize(d.grad)
        decoder1.zero_grad()
        decoder2.zero_grad()

    r_adv = d * eps

    # reopen bn, but freeze other params.
    # https://discuss.pytorch.org/t/why-is-it-when-i-call-require-grad-false-on-all-my-params-my-weights-in-the-network-would-still-update/22126/16
    decoder1.train()
    decoder2.train()
    return r_adv

class VATDecoderNetwork(nn.Module):
    def __init__(self, decoder_depth=1,
                decoder_embed_dim=512):

        super(VATDecoderNetwork, self).__init__()
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth

           # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0.) for i in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    def upsample_mask(self, mask, scale):
        assert len(mask.shape) == 2
        p = int(mask.shape[1] ** .5)
        return mask.reshape(-1, p, p).\
                    repeat_interleave(scale, axis=1).\
                    repeat_interleave(scale, axis=2)
    


    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w = x.shape
        mask = mask.reshape(-1, h, w).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 4:
            n, c, _, _ = pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)

        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, mask, fa=None, icg=None, data_shape=None, t_model=None):
        if t_model is not None:
            r_adv = get_r_adv_t(fa, icg, t_model[0], t_model[1], it=1, xi=1e-6, eps=2.0)
            fa += r_adv

        pred = self.forward_decoder(fa, mask)
        loss = self.forward_loss(fa, pred, mask)
        return loss, pred




class ASPP(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation_rates=(12, 24, 36),
                 hidden_channels=256,
                 norm_act=nn.BatchNorm2d,
                 pooling_size=None):
        super(ASPP, self).__init__()
        self.pooling_size = pooling_size

        self.map_convs = nn.ModuleList([
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[0],
                      padding=dilation_rates[0]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[1],
                      padding=dilation_rates[1]),
            nn.Conv2d(in_channels, hidden_channels, 3, bias=False, dilation=dilation_rates[2],
                      padding=dilation_rates[2])
        ])
        self.map_bn = norm_act(hidden_channels * 4)

        self.global_pooling_conv = nn.Conv2d(in_channels, hidden_channels, 1, bias=False)
        self.global_pooling_bn = norm_act(hidden_channels)

        self.red_conv = nn.Conv2d(hidden_channels * 4, out_channels, 1, bias=False)
        self.pool_red_conv = nn.Conv2d(hidden_channels, out_channels, 1, bias=False)
        self.red_bn = norm_act(out_channels)

        self.leak_relu = nn.LeakyReLU()

    def forward(self, x):
        # Map convolutions
        out = torch.cat([m(x) for m in self.map_convs], dim=1)
        out = self.map_bn(out)
        out = self.leak_relu(out)       # add activation layer
        out = self.red_conv(out)

        # Global pooling
        pool = self._global_pooling(x)
        pool = self.global_pooling_conv(pool)
        pool = self.global_pooling_bn(pool)
        pool = self.leak_relu(pool)  # add activation layer
        pool = self.pool_red_conv(pool)
        if self.training or self.pooling_size is None:
            pool = pool.repeat(1, 1, x.size(2), x.size(3))

        out += pool
        out = self.red_bn(out)
        out = self.leak_relu(out)  # add activation layer
        return out

    def _global_pooling(self, x):
        if self.training or self.pooling_size is None:
            pool = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
            pool = pool.view(x.size(0), x.size(1), 1, 1)
        else:
            raise NotImplementedError
        return pool


class Head(nn.Module):
    def __init__(self, classify_classes, norm_act=nn.BatchNorm2d, bn_momentum=0.0003):
        super(Head, self).__init__()

        self.classify_classes = classify_classes
        self.aspp = ASPP(2048, 256, [6, 12, 18], norm_act=norm_act)

        self.reduce = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False),
            norm_act(48, momentum=bn_momentum),
            nn.ReLU(),
        )
        self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       norm_act(256, momentum=bn_momentum),
                                       nn.ReLU(),
                                       )

    def forward(self, f_list):
        f = f_list[-1]
        f = self.aspp(f)
        low_level_features = f_list[0]
        low_h, low_w = low_level_features.size(2), low_level_features.size(3)
        low_level_features = self.reduce(low_level_features)
        f = F.interpolate(f, size=(low_h, low_w),
                          mode='bilinear', align_corners=True)
        f = torch.cat((f, low_level_features), dim=1)
        return f


