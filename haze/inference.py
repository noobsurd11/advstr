# 

#!/usr/bin/env python3
"""
inference_road.py

Realistic depth-aware fog generator for road scenes (same-resolution output).
Uses:
 - trained FogGenerator + TransmissionNet checkpoint (your model)
 - MiDaS_small (isl-org/MiDaS) for fast depth estimation
 - bilateral smoothing, per-pixel airlight, multi-scale noise
Designed for generating fog/haze for STR / adverse-weather driving datasets.
"""

import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

# -- import your modules (adjust names if different) --
from haze import SimpleGridNet, TransmissionNet, FogGenerator

# -------------------------
# Utils
# -------------------------
def load_midas_small(device):
    # fast small MiDaS
    model_name = "MiDaS_small"
    midas = torch.hub.load("isl-org/MiDaS", model_name, pretrained=True)
    midas.to(device).eval()
    transforms_midas = torch.hub.load("isl-org/MiDaS", "transforms")
    transform = transforms_midas.small_transform
    return midas, transform

def estimate_depth(midas, transform, device, img_bgr):
    # img_bgr: HxWx3 (uint8 BGR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        if isinstance(pred, dict): pred = pred["pred"]
    # pred: [1, H', W'] maybe. Resize to original.
    depth = F.interpolate(pred.unsqueeze(1), size=img_bgr.shape[:2], mode='bicubic', align_corners=False)
    depth = depth.squeeze().cpu().numpy()
    depth = depth - depth.min()
    if depth.max() > 0:
        depth = depth / (depth.max() + 1e-8)
    # Invert: MiDaS_small ~ inverse-depth (near=bright) --> we want far=1
    depth = 1.0 - depth
    # convert to float32
    return depth.astype(np.float32)

def smooth_depth(depth_np, d=9, sigma_color=0.05, sigma_space=9):
    # bilateralFilter expects uint8 or float32 in 0-1. Use float32.
    # OpenCV bilateral filter params: diameter, sigmaColor, sigmaSpace
    return cv2.bilateralFilter(depth_np, d, sigma_color*255.0, sigma_space)

def build_airlight_map(img_rgb, depth, sky_thresh=0.85):
    # img_rgb: HxWx3 in [0,1], depth: [H,W] in [0,1] where 1=far
    H, W = depth.shape
    # coarse sky detection: bright + high (near 1 in depth)
    gray = cv2.cvtColor((img_rgb*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)/255.0
    sky_mask = ((gray > 0.7) & (depth > sky_thresh)).astype(np.float32)
    # base A color: sample top few rows average color (where sky_mask high)
    top_rows = img_rgb[0:int(max(1, H*0.08)), :, :]
    A_base = top_rows.mean(axis=(0,1)) if top_rows.size>0 else np.array([0.9,0.9,0.95])
    # build low-frequency noise map and blend
    xv, yv = np.meshgrid(np.linspace(0,1,W), np.linspace(0,1,H))
    lowfreq = (np.sin(xv*3.14*2) + np.cos(yv*3.14*1.5))*0.5
    lowfreq = (lowfreq - lowfreq.min())/(lowfreq.max()-lowfreq.min()+1e-8)
    # A_map shape [H,W,3]
    A_map = np.ones((H,W,3), dtype=np.float32) * A_base.reshape(1,1,3)
    # boost airlight in sky regions and far regions
    boost = 0.15 * lowfreq[...,None] + 0.6*sky_mask[...,None] + 0.25*(depth[...,None])
    A_map = np.clip(A_map * (1.0 + boost*0.35), 0.0, 1.0)
    return A_map

def multi_scale_noise(H, W, device, scales=(64,16,4)):
    # produce three scaled noise maps, upsample and combine
    res = torch.zeros((1,1,H,W), device=device)
    for s, w in zip(scales, [0.5,0.35,0.15]):
        small = torch.randn(1,1,max(4, H//s), max(4, W//s), device=device)
        up = F.interpolate(small, size=(H,W), mode='bicubic', align_corners=False)
        res = res + w*up
    # normalize to [0,1]
    mn = res.min(); mx = res.max()
    if mx - mn > 1e-6:
        res = (res - mn) / (mx - mn)
    else:
        res = torch.zeros_like(res)
    return res  # [1,1,H,W]

# -------------------------
# Main pipeline
# -------------------------
def load_models(ckpt_path, device, alpha=0.1, beta_init=0.5):
    # instantiate model classes (must match training)
    backbone = SimpleGridNet(in_ch=3, base_ch=16)
    trans = TransmissionNet()
    fog = FogGenerator(backbone, alpha=alpha, beta_init=beta_init)
    ck = torch.load(ckpt_path, map_location='cpu')
    fog.load_state_dict(ck.get('fog_gen', ck), strict=False)
    trans.load_state_dict(ck.get('trans_net', ck), strict=False)
    fog.to(device).eval()
    trans.to(device).eval()
    return fog, trans

def inference_image(ckpt, img_path, out_path, preset='medium', beta=1.2, mix=0.7,
                    strength=1.0, keep_near=0.12, vertical_strength=0.35, add_noise=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # presets tune beta/strength/mix
    if preset == 'light':
        beta = 0.8 if beta is None else beta
        strength = 0.6 * strength
        mix = 0.6
    elif preset == 'medium':
        beta = 1.2 if beta is None else beta
        strength = 1.0 * strength
        mix = 0.75
    elif preset == 'heavy':
        beta = 1.8 if beta is None else beta
        strength = 1.6 * strength
        mix = 0.85

    fog_model, trans_net = load_models(ckpt, device)

    # load image (keep original resolution)
    pil = Image.open(img_path).convert('RGB')
    W, H = pil.size
    img_rgb = np.array(pil).astype(np.uint8)  # HxWx3 RGB
    img_bgr = img_rgb[..., ::-1]  # BGR for MiDaS

    # load MiDaS
    midas, midas_tfm = load_midas_small(device)
    depth = estimate_depth(midas, midas_tfm, device, img_bgr)  # HxW float32 (far=1)
    depth = smooth_depth(depth, d=9, sigma_color=0.08, sigma_space=9)

    # ensure depth in [0,1]
    depth = np.clip(depth, 0.0, 1.0)

    # build A_map
    img_rgb_f = img_rgb.astype(np.float32)/255.0
    A_map = build_airlight_map(img_rgb_f, depth)

    # prepare tensor input
    to_tensor = transforms.ToTensor()
    J = to_tensor(pil).unsqueeze(0).to(device)  # [1,3,H,W], values [0,1]

    with torch.no_grad():
        t_pred = trans_net(J)  # [1,1,H',W'] probably equals H,W if net is fully conv
        # If t_pred spatial dims differ, resize
        t_pred = F.interpolate(t_pred, size=(H, W), mode='bilinear', align_corners=False)

    # depth transmission
    t_depth = np.exp(-beta * depth)  # [H,W], far -> small t (foggy) ? Wait: depth is far=1 -> exp(-beta*1) small -> correct
    # we want far => lower transmission, so t_depth = exp(-beta * depth) is correct (depth large => small t)

    # convert to torch
    t_depth_t = torch.from_numpy(t_depth).float().unsqueeze(0).unsqueeze(0).to(device)

    # combine: t_final = mix * t_depth + (1-mix) * t_pred
    # but we want the model t_pred to guide small-scale structure; depth dominates coarse attenuation
    t_final = mix * t_depth_t + (1.0 - mix) * t_pred
    # ensure not too small near camera
    t_final = torch.clamp(t_final, keep_near, 1.0)

    # vertical accumulation (more haze at upper rows / far)
    B,_,h_t,w_t = t_final.shape
    Gy = torch.linspace(1.0, 0.85, steps=h_t, device=device).view(1,1,h_t,1).expand(B,1,h_t,w_t)
    t_final = t_final * (1.0 - vertical_strength * (1.0 - Gy))

    # synthesize using FogGenerator (keeps fog texture & alpha)
    with torch.no_grad():
        I_pred, fog_feat, A_param = fog_model(J, t_final)  # I_pred in [0,1]

    # convert A_map to torch
    A_map_t = torch.from_numpy(A_map.transpose(2,0,1)).float().unsqueeze(0).to(device)  # [1,3,H,W]

    # Replace global A with spatial A_map influence: blend predicted I with A_map
    # Compute physical composite: I_phys = J * t + A_map * (1 - t)
    t3 = t_final.expand(-1,3,-1,-1)
    I_phys = J * t3 + A_map_t * (1.0 - t3)

    # Add fog texture: small scaled multi-scale noise that follows (1-t)
    if add_noise:
        noise = multi_scale_noise(H, W, device, scales=(64,16,4))  # [1,1,H,W]
        # broadcast to 3 channels and scale by strength*(1-t)
        noise_col = noise.expand(1,3,H,W)
        I_final = I_phys + 0.15 * strength * (1.0 - t3) * noise_col
    else:
        I_final = I_phys

    # clip and convert to numpy
    I_final = torch.clamp(I_final, 0.0, 1.0).squeeze(0).permute(1,2,0).detach().cpu().numpy()
    I_final = (I_final * 255.0).astype(np.uint8)

    # Optional mild tone mapping: reduce oversaturation in highlights
    I_out = cv2.cvtColor(I_final, cv2.COLOR_RGB2BGR)
    # slight gamma correction for realism
    gamma = 1.02
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
    I_out = cv2.LUT(I_out, table)

    Image.fromarray(cv2.cvtColor(I_out, cv2.COLOR_BGR2RGB)).save(out_path)
    print("Saved:", out_path)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to checkpoint (best_model.pth)")
    p.add_argument("--img", required=True, help="input image (RGB)")
    p.add_argument("--out", required=True, help="output path")
    p.add_argument("--preset", default="medium", choices=["light","medium","heavy"])
    p.add_argument("--beta", type=float, default=None, help="base beta for depth attenuation (override preset)")
    p.add_argument("--mix", type=float, default=0.75, help="mix factor: how much depth vs model t (0..1)")
    p.add_argument("--strength", type=float, default=1.0, help="overall fog strength multiplier")
    p.add_argument("--keep_near", type=float, default=0.12, help="minimum transmission near camera (0..1)")
    p.add_argument("--vertical_strength", type=float, default=0.35, help="vertical accumulation strength")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    inference_image(args.ckpt, args.img, args.out, preset=args.preset, beta=args.beta,
                    mix=args.mix, strength=args.strength, keep_near=args.keep_near,
                    vertical_strength=args.vertical_strength)
