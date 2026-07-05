import torch
import torch.nn as nn
import torch.nn.functional as F


#  Basic Blocks 

class ConvBNSiLU(nn.Module):
    def __init__(self, ic, oc, k=1, s=1):
        super().__init__()
        self.conv = nn.Conv2d(ic, oc, k, s, k // 2, bias=False)
        self.bn   = nn.BatchNorm2d(oc)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = ConvBNSiLU(c, c // 2)
        self.cv2 = ConvBNSiLU(c // 2, c, 3)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class CSPLayer(nn.Module):
    def __init__(self, ic, oc, n=1):
        super().__init__()
        h        = oc // 2
        self.cv1 = ConvBNSiLU(ic, h)
        self.cv2 = ConvBNSiLU(ic, h)
        self.cv3 = ConvBNSiLU(2 * h, oc)
        self.m   = nn.Sequential(*[Bottleneck(h) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat([self.m(self.cv1(x)), self.cv2(x)], 1))


#  Backbone 

class Focus(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv = ConvBNSiLU(ic * 4, oc, 3)

    def forward(self, x):
        return self.conv(torch.cat([
            x[..., ::2,  ::2],
            x[..., 1::2, ::2],
            x[..., ::2,  1::2],
            x[..., 1::2, 1::2],
        ], 1))


class CSPDarknet(nn.Module):
    CONFIGS = {'s': (0.50, 0.33), 'm': (0.75, 0.67), 'l': (1.0, 1.0)}

    def __init__(self, size='s'):
        super().__init__()
        w, d = self.CONFIGS[size]

        def ch(c): return max(1, int(c * w))
        def dp(n): return max(1, round(n * d))

        self.stem  = Focus(3, ch(64))
        self.dark2 = nn.Sequential(ConvBNSiLU(ch(64),   ch(128),  3, 2), CSPLayer(ch(128),  ch(128),  dp(3)))
        self.dark3 = nn.Sequential(ConvBNSiLU(ch(128),  ch(256),  3, 2), CSPLayer(ch(256),  ch(256),  dp(9)))
        self.dark4 = nn.Sequential(ConvBNSiLU(ch(256),  ch(512),  3, 2), CSPLayer(ch(512),  ch(512),  dp(9)))
        self.dark5 = nn.Sequential(ConvBNSiLU(ch(512),  ch(1024), 3, 2), CSPLayer(ch(1024), ch(1024), dp(3)))
        self.out_channels = (ch(256), ch(512), ch(1024))

    def forward(self, x):
        x  = self.stem(x)
        x  = self.dark2(x)
        p3 = self.dark3(x)
        p4 = self.dark4(p3)
        p5 = self.dark5(p4)
        return p3, p4, p5


#  CFE Module 

class FFMBlock(nn.Module):
    def __init__(self, c3, c4, c5, out_c):
        super().__init__()
        self.conv = nn.Conv2d(c3 + c4 + c5, out_c, 3, 1, 1, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)

    def forward(self, p3, p4, p5):
        h, w = p3.shape[2:]
        cf = torch.cat([
            F.interpolate(p5, (h, w), mode='nearest'),
            F.interpolate(p4, (h, w), mode='nearest'),
            p3,
        ], 1)
        x = self.bn(self.conv(cf))
        return x * torch.sigmoid(x)   # Swish / SiLU


class InjectBlock(nn.Module):
    def __init__(self, pn_c, pf_c, oc):
        super().__init__()
        self.cv_pn = nn.Conv2d(pn_c, oc, 1, bias=False)
        self.cv_pf = nn.Conv2d(pf_c, oc, 1, bias=False)

    def forward(self, pn, pf):
        h, w   = pn.shape[2:]
        pf_r   = F.interpolate(pf, (h, w), mode='nearest')
        attn   = torch.sigmoid(self.cv_pf(pf_r))
        scaled = self.cv_pn(pn) * F.adaptive_avg_pool2d(attn, (h, w))
        resid  = F.adaptive_avg_pool2d(self.cv_pf(pf_r), (h, w))
        return scaled + resid


class CFEModule(nn.Module):
    def __init__(self, c3, c4, c5, pf_c=256):
        super().__init__()
        self.ffm  = FFMBlock(c3, c4, c5, pf_c)
        self.inj3 = InjectBlock(c3, pf_c, c3)
        self.inj4 = InjectBlock(c4, pf_c, c4)
        self.inj5 = InjectBlock(c5, pf_c, c5)
        self.csp3 = CSPLayer(c3,      c3, 3)
        self.csp4 = CSPLayer(c3 + c4, c4, 3)
        self.csp5 = CSPLayer(c4 + c5, c5, 3)

    def forward(self, p3, p4, p5):
        pf   = self.ffm(p3, p4, p5)
        p3p  = self.inj3(p3, pf)
        p4p  = self.inj4(p4, pf)
        p5p  = self.inj5(p5, pf)
        out3 = self.csp3(p3p)
        out4 = self.csp4(torch.cat([p4p, F.max_pool2d(out3, 2)], 1))
        out5 = self.csp5(torch.cat([p5p, F.max_pool2d(out4, 2)], 1))
        return out3, out4, out5


#  Detection Head 

class YoloHead(nn.Module):
    def __init__(self, channels, nc=1):
        super().__init__()
        self.cls = nn.ModuleList()
        self.reg = nn.ModuleList()
        self.obj = nn.ModuleList()
        for c in channels:
            self.cls.append(nn.Sequential(ConvBNSiLU(c, c, 3), ConvBNSiLU(c, c, 3), nn.Conv2d(c, nc, 1)))
            self.reg.append(nn.Sequential(ConvBNSiLU(c, c, 3), ConvBNSiLU(c, c, 3), nn.Conv2d(c, 4,  1)))
            self.obj.append(nn.Sequential(ConvBNSiLU(c, c, 3), ConvBNSiLU(c, c, 3), nn.Conv2d(c, 1,  1)))

    def forward(self, feats):
        return [
            torch.cat([self.reg[i](f), self.obj[i](f), self.cls[i](f)], 1)
            for i, f in enumerate(feats)
        ]


#  Full Model 

class YoloXCFE(nn.Module):
    def __init__(self, size='s', nc=1):
        super().__init__()
        self.backbone = CSPDarknet(size)
        c3, c4, c5   = self.backbone.out_channels
        self.cfe      = CFEModule(c3, c4, c5, max(128, c3 // 2))
        self.head     = YoloHead([c3, c4, c5], nc)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        o3, o4, o5 = self.cfe(p3, p4, p5)
        return self.head([o3, o4, o5])