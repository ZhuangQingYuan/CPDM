from Lib.SegModels.Backbone.BasedOnConvolution import resnet
from Lib.SegModels.Backbone.BasedOnConvolution.xception import xception
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter

from Lib.Tools.SELFNET.origin_prototype import distributed_sinkhorn, momentum_update, l2_normalize
from einops import rearrange, repeat


class SELFNET(nn.Module):
    def __init__(self, cfg):
        super(SELFNET, self).__init__()
        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True, replace_stride_with_dilation=cfg[
                'replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048
        self.num_classes = cfg['num_classes']
        self.head = ASPPModule(high_channels, cfg['dilations'])
        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))
        self.num_prototype = 3

        self.classifier = nn.Conv2d(256, self.num_classes, 1, bias=True)

        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_prototype, 256),requires_grad=True)

        self.proj_head = ProjHead(256,256)

        self.feat_norm = nn.LayerNorm(256)

        self.mask_norm = nn.LayerNorm(self.num_classes)



    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)
        c1 = self.reduce(c1)
        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        out = self.classifier(feature)
        return out,feature

    def prototype_learning(self, _c, out_seg, gt_seg, masks):
        pred_seg = torch.max(out_seg, 1)[1]
        mask = (gt_seg == pred_seg.view(-1))
        cosine_similarity = torch.mm(_c, self.prototypes.view(-1, self.prototypes.shape[-1]).t())
        proto_logits = cosine_similarity
        proto_target = gt_seg.clone().float()
        # clustering for each class
        protos = self.prototypes.data.clone()
        for k in range(self.num_classes):
            init_q = masks[..., k]
            init_q = init_q[gt_seg == k, ...]
            if init_q.shape[0] == 0:
                continue
            q, indexs = distributed_sinkhorn(init_q)
            m_k = mask[gt_seg == k]
            c_k = _c[gt_seg == k, ...]
            m_k_tile = repeat(m_k, 'n -> n tile', tile=self.num_prototype)
            m_q = q * m_k_tile  # n x self.num_prototype
            c_k_tile = repeat(m_k, 'n -> n tile', tile=c_k.shape[-1])
            c_q = c_k * c_k_tile  # n x embedding_dim
            f = m_q.transpose(0, 1) @ c_q  # self.num_prototype x embedding_dim
            n = torch.sum(m_q, dim=0)
            if torch.sum(n) > 0:
                f = F.normalize(f, p=2, dim=-1)
                new_value = momentum_update(old_value=protos[k, n != 0, :], new_value=f[n != 0, :], momentum=0.999,debug=False)
                protos[k, n != 0, :] = new_value
            proto_target[gt_seg == k] = indexs.float() + (self.num_prototype * k)
        self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
        return proto_logits, proto_target

    def forward(self, x, **kwargs):
        dict_return = {}
        mode = kwargs.get('mode', None)
        need_fp = kwargs.get('need_fp', None)
        need_prototype = kwargs.get('need_prototype', None)
        gt_seg = kwargs.get('gt_seg', None)

        h, w = x.shape[-2:]
        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]
        out,feature = self._decode(c1, c4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        dict_return['out'] = out
        if mode == "train":
            if need_fp:
                out_fp,_ = self._decode(nn.Dropout2d(0.5)(c1), nn.Dropout2d(0.5)(c4))
                out_fp = F.interpolate(out_fp, size=(h, w), mode="bilinear", align_corners=True)
                dict_return['out_fp'] = out_fp
            if need_prototype:
                # 通过proj映射后进行的原型学习分割结果该结果应该和没用之前的类似 一致性学习
                c = self.proj_head(feature)
                c = F.interpolate(c, size=(h, w), mode="bilinear", align_corners=True)
                _c = rearrange(c, 'b c h w -> (b h w) c')
                _c = self.feat_norm(_c)
                _c = l2_normalize(_c)
                self.prototypes.data.copy_(l2_normalize(self.prototypes))
                masks = torch.einsum('nd,kmd->nmk', _c, self.prototypes)
                out_proto_seg = torch.amax(masks, dim=1)
                out_proto_seg = self.mask_norm(out_proto_seg)
                out_proto_seg = rearrange(out_proto_seg, "(b h w) k -> b k h w", b=c.shape[0], h=c.shape[2])
                out_proto_seg_l,out_proto_seg_u = out_proto_seg.chunk(2)
                _c_l,_c_u = _c.chunk(2)
                masks_l,masks_u = masks.chunk(2)
                if gt_seg is None:
                    raise KeyError("必须传入gt_seg|原型更新需要原始标签信息")
                else:
                    gt_seg = gt_seg.float().view(-1)
                    contrast_logits, contrast_target = self.prototype_learning(_c_l, out_proto_seg_l, gt_seg, masks_l)
                    dict_return['contrast_logits'] = contrast_logits
                    dict_return['contrast_target'] = contrast_target
                dict_return['out_proto_seg_l'] = out_proto_seg_l
                dict_return['out_proto_seg_u'] = out_proto_seg_u
        if mode == "eval":
            dict_return['mid_features'] = feature
            pass
        return dict_return


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)

class ProjHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256):
        super(ProjHead, self).__init__()

        self.proj = self.mlp2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_in, proj_dim, 1))

    def forward(self, x):
        return l2_normalize(self.proj(x))
