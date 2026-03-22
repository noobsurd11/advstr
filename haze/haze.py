import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Residual Dense Block (simple)
# -------------------------
class RDB(nn.Module):
    def __init__(self, in_channels, growth=16, n_layers=4):
        super().__init__()
        layers = []
        in_ch = in_channels
        for i in range(n_layers):
            layers.append(nn.Conv2d(in_ch, growth, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_ch = growth
        self.body = nn.Sequential(*layers)
        self.lff = nn.Conv2d(growth, in_channels, 1)  # local feature fusion

    def forward(self, x):
        out = self.body(x)
        out = self.lff(out)
        return out + x


# -------------------------
# Simple Grid-like backbone (inspired by GridNet)
# height=3, width=6 variant-ish
# -------------------------
class SimpleGridNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=16, rdb_layers=4):
        super().__init__()
        self.conv_in = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        # three scale streams (h=3)
        # stream 0: full-res
        # stream 1: downsample x2
        # stream 2: downsample x4
        self.rdb00 = RDB(base_ch, growth=16, n_layers=rdb_layers)
        self.down01 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.rdb10 = RDB(base_ch, growth=16, n_layers=rdb_layers)
        self.down12 = nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1)
        self.rdb20 = RDB(base_ch, growth=16, n_layers=rdb_layers)

        # upsample paths
        self.up21 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.up10 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)

        # some merges
        self.merge01 = nn.Conv2d(base_ch*2, base_ch, 1)
        self.merge12 = nn.Conv2d(base_ch*2, base_ch, 1)

        # final conv to produce features
        self.conv_out = nn.Conv2d(base_ch, base_ch, 3, padding=1)
        self.out_channels = base_ch

    def forward(self, x):
        x0 = self.conv_in(x)              # full-res
        r0 = self.rdb00(x0)

        x1 = self.down01(r0)
        r1 = self.rdb10(x1)

        x2 = self.down12(r1)
        r2 = self.rdb20(x2)

        # upsample r2->r1
        u1 = self.up21(r2)
        m1 = torch.cat([u1, r1], dim=1)
        m1 = self.merge12(m1)

        # upsample m1->r0
        u0 = self.up10(m1)
        m0 = torch.cat([u0, r0], dim=1)
        m0 = self.merge01(m0)

        out = self.conv_out(m0)
        return out


# -------------------------
# Transmission Network
# -------------------------
class TransmissionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Fog Generator (GridHazeNet)
# -------------------------
class FogGenerator(nn.Module):
    def __init__(self, backbone: nn.Module, alpha=0.1, beta_init=0.5):
        super().__init__()
        self.backbone = backbone
        feat_ch = backbone.out_channels
        self.conv_out = nn.Conv2d(feat_ch, 3, 3, padding=1)  # produce fog feature map
        self.alpha = alpha
        # learnable A base + var and beta
        self.A_base = nn.Parameter(torch.tensor([0.8,0.8,0.9], dtype=torch.float32))
        self.A_var  = nn.Parameter(torch.tensor([0.1,0.1,0.1], dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(float(beta_init)))

    def forward(self, J, t):  # J: clear rgb, t: [B,1,H,W]
        feat = self.backbone(J)                # [B, C, H, W]
        F = self.conv_out(feat)                # [B,3,H,W]
        fog = self.beta * F

        B,_,H,W = J.shape
        A = torch.tanh(self.A_base + 0.5*self.A_var).view(1,3,1,1).expand(B,3,H,W)

        t3 = t.expand(-1,3,-1,-1)
        I_pred = J * t3 + A * (1 - t3) + self.alpha * fog
        return I_pred, F, A
