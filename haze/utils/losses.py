import torch
import torch.nn.functional as F
from torchvision import models

# Dark channel helper
def dark_channel(x, patch_size=15):
    # x: [B,3,H,W]
    min_rgb = torch.min(x, dim=1, keepdim=True)[0]
    pad = patch_size // 2
    neg = -min_rgb
    out = -F.max_pool2d(neg, kernel_size=patch_size, stride=1, padding=pad)
    return out

def dark_channel_loss(I_pred, I_gt, patch_size=15):
    dc_pred = dark_channel(I_pred, patch_size)
    dc_gt = dark_channel(I_gt, patch_size)
    return F.mse_loss(dc_pred, dc_gt)

# Perceptual loss using VGG16 features upto layer index
class PerceptualLoss(torch.nn.Module):
    def __init__(self, layer_idx=16, device='cpu'):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features
        self.vgg = torch.nn.Sequential(*list(vgg.children())[:layer_idx]).to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        # normalizing constants for VGG
        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x, y):
        # x,y in [0,1]
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        fx = self.vgg(x)
        fy = self.vgg(y)
        return F.mse_loss(fx, fy)
