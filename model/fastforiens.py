import torch
import torch.nn as nn
import itertools
import torch.nn.functional as F
from timm.models.layers import SqueezeExcite
# from model.utils.torch_wavelets import DWT_2D, IDWT_2D
from pytorch_wavelets import DWT, IDWT


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        x[0] = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x[0]))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if isinstance(self.m, LocalWindowAttention):
            if self.training and self.drop > 0:
                x[0] = x[0] + self.m(x) * torch.rand(x[0].size(0), 1, 1, 1,
                                                  device=x[0].device).ge_(self.drop).div(1 - self.drop).detach()
                return x
            else:
                x[0] = x[0] + self.m(x)[0]
                return x
        if self.training and self.drop > 0:
            x[0] = x[0] + self.m(x[0]) * torch.rand(x[0].size(0), 1, 1, 1,
                                             device=x[0].device).ge_(self.drop).div(1 - self.drop).detach()
            return x
        else:
            x[0] = x[0] + self.m(x[0])
            return x


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
        query (tensor): 一个全局的Q
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 kernels=[5, 5, 5, 5],
                 ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        for i in range(num_heads):
            # qkv少一个生成Q的dim维度
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim + self.d, resolution=resolution))
        self.qs = torch.nn.Sequential(
            Conv2d_BN(dim, self.key_dim, resolution=resolution)
        )
        self.dws = Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                             resolution=resolution)

        self.channel_interaction = nn.Sequential(
            nn.Conv2d(128, self.d, kernel_size=1),
            nn.BatchNorm2d(self.d),
            nn.GELU(),
        )
        self.channel_interaction_fre = nn.Sequential(
            nn.Conv2d(128, self.d, kernel_size=1),
            nn.BatchNorm2d(self.d),
            nn.GELU(),
        )
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        # Use of parameters during training
        # self.dwt = DWT(wave='haar').to(torch.float16)
        # self.idwt = IDWT(wave='haar').to(torch.float16)

        # Use of parameters during testing
        self.dwt = DWT(wave='haar')
        self.idwt = IDWT(wave='haar')

        self.reduce = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 16, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(inplace=True),
        )
        self.reduce_one = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 16, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(inplace=True),
        )

        self.reduce_v = nn.Sequential(
            nn.Conv2d(128, 128 // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm2d(128 // 4),
            nn.ReLU(inplace=True),
        )

        self.filter = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )

        self.filter_one = nn.Sequential(
            nn.Conv2d(dim // 4, dim // 4, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.dim = dim
        if dim == 128:
            self.linear_dwt = nn.Linear(4*4, 7*7, bias=False)
        elif dim == 256:
            self.linear_dwt = nn.Linear(4*4, 7*7, bias=False)
        elif dim == 384:
            self.linear_dwt = nn.Linear(2*2, 4*4, bias=False)

        self.proj_concat = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim // 4 + dim // 16, dim // 4 , bn_weight_init=0, resolution=resolution))
        self.proj_concat_one = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            dim // 4 + dim // 16, dim // 4, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x, x_c, x_):  # x (B,C,H,W)
        B, C, H, W = x.shape
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        feats_in = x.chunk(len(self.qkvs), dim=1)
        # feats_in_fre = x_.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = feats_in[0]
        # globel query
        q_ = self.qs(x)
        q_ = q_.flatten(2)
        # globel value frequery
        x_temp = self.reduce_v(x_c)
        y = self.dwt(x_temp)
        y_ll = y[0]
        y_lh = y[1][0][:, :, 0, :, :]
        y_hl = y[1][0][:, :, 1, :, :]
        y_hh = y[1][0][:, :, 2, :, :]
        v_dwt = torch.cat([y_ll, y_lh, y_hl, y_hh], dim=1)
        v_dwt = self.channel_interaction_fre(torch.nn.functional.adaptive_avg_pool2d(v_dwt, 1)).flatten(2)
        # globel value RGB
        v_ = self.channel_interaction(torch.nn.functional.adaptive_avg_pool2d(x_c, 1)).flatten(2)

        for i, qkv in enumerate(self.qkvs):
            if i == 0 or i == 2:
                if i > 0:  # add the previous output to the input
                    feat = feat + feats_in[i]
                feat = qkv(feat)
                k, v = feat.view(B, -1, H, W).split([self.key_dim, self.d], dim=1)  # B, C/h, H, W
                # q_singal = self.dws[i](q_singal)
                k, v = k.flatten(2), v.flatten(2)  # B, C/h, N
                v = torch.sigmoid(v_) * v
                # q_ = q + q_
                attn = (
                        (q_.transpose(-2, -1) @ k) * self.scale
                        +
                        (trainingab[i] if self.training else self.ab[i])
                )
                attn = attn.softmax(dim=-1)  # BNN
                feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
                feats_out.append(feat)
            else:
                if i > 0:  # add the previous output to the input
                    feat = feat + feats_in[i]
                # local frequery input
                if i == 1:
                    y = self.dwt(self.reduce(feat))
                else:
                    y = self.dwt(self.reduce_one(feat))
                y_ll = y[0]
                y_lh = y[1][0][:, :, 0, :, :]
                y_hl = y[1][0][:, :, 1, :, :]
                y_hh = y[1][0][:, :, 2, :, :]
                x_dwt = torch.cat([y_ll, y_lh, y_hl, y_hh], dim=1)
                # IDWT CONCAT INPUT
                if i == 1:
                    x_dwt = self.filter(x_dwt)
                else:
                    x_dwt = self.filter_one(x_dwt)
                x_ll = x_dwt[:, :(self.dim // 16), :, :]
                x_lh = x_dwt[:, (self.dim // 16):((self.dim // 16)*2), :, :].unsqueeze(dim=2)
                x_hl = x_dwt[:, ((self.dim // 16)*2):((self.dim // 16)*3), :, :].unsqueeze(dim=2)
                x_hh = x_dwt[:, ((self.dim // 16)*3):((self.dim // 16)*4), :, :].unsqueeze(dim=2)
                x_all = torch.cat([x_lh, x_hl, x_hh],dim=2)
                x_all_ = (x_ll, [x_all])
                feat_idwt = F.interpolate(self.idwt(x_all_), size=[feat.shape[2], feat.shape[3]])

                feat_dwt = qkv(self.linear_dwt(x_dwt.flatten(2)).view(B, -1, H, W))

                k, v = feat_dwt.view(B, -1, H, W).split([self.key_dim, self.d], dim=1)  # B, C/h, H, W
                # q_singal = self.dws[i](q_singal)
                k, v = k.flatten(2), v.flatten(2)  # B, C/h, N
                v = torch.sigmoid(v_dwt) * v
                # q_ = q + q_
                attn = (
                        (q_.transpose(-2, -1) @ k) * self.scale
                        +
                        (trainingab[i] if self.training else self.ab[i])
                )
                attn = attn.softmax(dim=-1)  # BNN
                feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW

                if i == 1:
                    feats_out.append(self.proj_concat(torch.cat([feat, feat_idwt], dim=1)))
                else:
                    feats_out.append(self.proj_concat_one(torch.cat([feat, feat_idwt], dim=1)))

        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],

                 ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        window_resolution = min(window_resolution, resolution)
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                           attn_ratio=attn_ratio,
                                           resolution=window_resolution,
                                           kernels=kernels, )


    def forward(self, x):
        H = W = self.resolution
        B, C, H_, W_ = x[0].shape
        _, C_c, H_c, W_c = x[1].shape
        x_ = x[0]
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))
        # x_mutil_scale = self.pappm(x)

        if H <= self.window_resolution and W <= self.window_resolution:
            x[0] = self.attn(x[0], x[1], x_)
        else:
            x[0] = x[0].permute(0, 2, 3, 1)
            # x_mutil_scale = x_mutil_scale.permute(0, 2, 3, 1)

            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x[0] = torch.nn.functional.pad(x[0], (0, 0, 0, pad_r, 0, pad_b))
                # x_mutil_scale = torch.nn.functional.pad(x_mutil_scale, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x[0] = x[0].view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x[1] = x[1].view(B, nH, self.window_resolution * 2, nW, self.window_resolution * 2, C_c).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution * 2, self.window_resolution * 2, C_c
            ).permute(0, 3, 1, 2)

            x[0] = self.attn(x[0], x[1], x_)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x[0] = x[0].permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                                           C).transpose(2, 3).reshape(B, pH, pW, C)
            x[1] = x[1].permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution * 2, self.window_resolution * 2,
                                                 C_c).transpose(2, 3).reshape(B, pH * 2, pW * 2, C_c)
            if padding:
                x[0] = x[0][:, :H, :W].contiguous()
            x[0] = x[0].permute(0, 3, 1, 2)
            x[1] = x[1].permute(0, 3, 1, 2)
        return x


class EfficientViTBlock(torch.nn.Module):
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """

    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=14,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()

        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                                                       resolution=resolution, window_resolution=window_resolution,
                                                       kernels=kernels))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        x = self.dw0(x)
        x = self.ffn0(x)
        x = self.mixer(x)
        x = self.dw1(x)
        x = self.ffn1(x)
        return x
        # return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None,
                 no_relu=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out if self.no_relu else self.relu(out)


class SegmentHead(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=False):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Dropout(0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )
        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class aera_Head(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=False):
        super(aera_Head, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1, padding=1)
        out_chan = n_classes
        self.conv_out = nn.Conv2d(mid_chan, out_chan, 1, 1, 0, bias=True)
        self.down = torch.nn.AdaptiveAvgPool2d(output_size=1)
        self.init_weight()
        self.m = torch.nn.Sigmoid()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.conv_out(feat)
        feat = self.down(feat)
        return self.m(feat)


class EiffVit_seg(nn.Module):
    def __init__(self,
                 num_classes,
                 base_channels=64,
                 patch_size=16,
                 img_size=224,
                 stages=['s', 's', 's'],
                 embed_dim=[128, 256, 384],
                 key_dim=[16, 16, 16],
                 depth=[1, 1, 1],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 layer_nums=[2, 2, 2, 2],
                 in_channels=3):
        super(EiffVit_seg, self).__init__()
        self.base_channels = base_channels
        base_chs = base_channels

        self.conv1 = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, base_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(),
            nn.Conv2d(
                base_chs, base_chs, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(),)
        self.layer1 = self._make_layer(BasicBlock, base_chs, base_chs, layer_nums[0])
        self.relu = nn.ReLU()
        self.layer2 = self._make_layer(BasicBlock, base_chs, base_chs * 2, layer_nums[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2, layer_nums[2], stride=2)

        # Transformer Block
        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                               Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)), ))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1],
                              resolution=resolution)),
                                               Residual(
                                                   FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)), ))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        self.compression = nn.Sequential(
            nn.BatchNorm2d(base_chs * 4),
            nn.ReLU(),
            nn.Conv2d(base_chs * 4, base_chs * 2, kernel_size=1)
        )

        self.layer3_ = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2, 1)

        self.layer_c1 = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2, 1)
        self.layer_c2 = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2, 1)
        self.layer_c3 = self._make_layer(BasicBlock, base_chs * 2, base_chs * 2, 1)

        self.diff1 = nn.Sequential(
            nn.Conv2d(base_chs * 4, base_chs * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_chs * 2, momentum=0.1)
        )
        self.diff2 = nn.Sequential(
            nn.Conv2d(base_chs * 6, base_chs * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_chs * 2, momentum=0.1)
        )
        self.head = SegmentHead(in_chan=base_chs * 2, mid_chan=base_chs * 2, n_classes=num_classes,)

        # Loss
        # self.bd_head = SegmentHead(in_chan=base_chs * 2, mid_chan=32, n_classes=1,)
        # self.area_head = aera_Head(in_chan=384, mid_chan=128,n_classes=1)
        # self.offset_head = SegmentHead(in_chan=base_chs * 2, mid_chan=base_chs * 2, n_classes=num_classes,)

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))

        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            if i == (blocks - 1):
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=True))
            else:
                layers.append(
                    block(
                        out_channels, out_channels, stride=1, no_relu=False))

        return nn.Sequential(*layers)

    def forward(self, x):
        width_output = x.shape[-1] // 8
        height_output = x.shape[-2] // 8

        x1 = self.layer1(self.conv1(x))  # c, 1/4
        x2 = self.layer2(self.relu(x1))  # 2c, 1/8
        x3 = self.layer3(self.relu(x2))  # 2c, 1/16

        # cognitive branch
        x_c_1 = self.layer_c1(x2)
        x3_1 = self.blocks1([x3, x_c_1])
        x_c_1 = x_c_1 + F.interpolate(x3_1[0], size=[height_output, width_output], mode='bilinear',
                                      align_corners=False)

        x_c_2 = self.layer_c2(x_c_1)
        x3_2 = self.blocks2([x3_1[0], x_c_2])
        x_c_2 = x_c_2 + F.interpolate(self.diff1(x3_2[0]), size=[height_output, width_output], mode='bilinear',
                                      align_corners=False)

        x_c_3 = self.layer_c3(x_c_2)
        x3_3 = self.blocks3([x3_2[0], x_c_3])
        x_c_3 = x_c_3 + F.interpolate(self.diff2(x3_3[0]), size=[height_output, width_output], mode='bilinear',
                                      align_corners=False)

        logit = self.head(x_c_3)

        return logit

# model = EiffVit_seg(num_classes=2)
# model.eval()
# input = torch.randn(size=[1, 3, 224, 224])
# output = model(input)