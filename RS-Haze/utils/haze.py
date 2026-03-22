# tools/synthesize_haze.py
import os, json, math, random
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# --------- DPT depth (MiDaS) setup ----------
# pip install timm torchvision
# (Use small model for speed; change to dpt_hybrid or large if you want)
import torchvision.transforms as T
from PIL import Image

def load_midas(model_name="DPT_Small"):
    import torch.hub as hub
    midas = hub.load("intel-isl/MiDaS", model_name)
    midas.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas.to(device)
    transforms = hub.load("intel-isl/MiDaS", "transforms")
    if "DPT" in model_name:
        transform = transforms.dpt_transform
    else:
        transform = transforms.small_transform
    return midas, transform, device

def estimate_depth(midas, transform, device, img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = transform(img).to(device)
    with torch.no_grad():
        pred = midas(inp)
        if isinstance(pred, dict): pred = pred["pred"]
        depth = F.interpolate(pred.unsqueeze(1), size=img_bgr.shape[:2], mode="bicubic", align_corners=False)
    depth = depth.squeeze().cpu().numpy()
    # normalize to [0,1] (near=0, far=1)
    d = depth - depth.min()
    d = d / (d.max() + 1e-8)
    return d

# --------- Non-homogeneous beta map ----------
def sample_beta_map(h, w, base_beta=(0.6,1.6)):
    # base globally
    beta0 = random.uniform(*base_beta)
    # spatially-varying component via low-freq noise
    grid = (np.random.randn(h//32+1, w//32+1)).astype(np.float32)
    grid = cv2.resize(grid, (w, h), interpolation=cv2.INTER_CUBIC)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)
    grid = 0.6*grid + 0.7  # scale to ~[0.7,1.3]
    beta = beta0 * grid
    return beta

def random_airlight():
    # slightly bluish/gray airlight
    base = np.array([1.0, 1.0, 1.0], np.float32)
    tint = np.array([random.uniform(0.92,1.00), random.uniform(0.95,1.00), 1.0], np.float32)
    A = (base * tint) * random.uniform(0.85, 1.0)
    return A  # in [0,1], 3-ch

# --------- Text-aware clamping inside boxes ----------
def build_text_mask(h, w, anns_for_img):
    mask = np.zeros((h,w), np.float32)
    for a in anns_for_img:
        if "bbox" in a:
            x,y,ww,hh = a["bbox"]
            x1,y1 = int(x), int(y)
            x2,y2 = int(x+ww), int(y+hh)
            x1 = max(0,x1); y1 = max(0,y1); x2 = min(w,x2); y2 = min(h,y2)
            mask[y1:y2, x1:x2] = 1.0
    return mask

def synthesize_haze(img_bgr, anns=None, keep_text_readable=True):
    H,W = img_bgr.shape[:2]
    # normalize to [0,1]
    J = (img_bgr.astype(np.float32)/255.0).clip(0,1)

    # depth
    global _MIDAS
    midas, transform, device = _MIDAS
    d = estimate_depth(midas, transform, device, img_bgr)  # [0,1], far=1

    # beta map & airlight
    beta = sample_beta_map(H,W, base_beta=(0.6,1.6))
    A = random_airlight()  # shape (3,)

    # transmission
    t = np.exp(-beta * d)  # [~e^-1.6, 1]

    # Optional: keep text readable
    if keep_text_readable and anns:
        text_mask = build_text_mask(H,W, anns)  # 1 inside boxes
        # soften strong haze within text boxes
        # enforce a floor on transmission inside text regions (e.g., >= 0.55–0.75)
        t_min = random.uniform(0.55, 0.75)
        t = np.where(text_mask>0, np.maximum(t, t_min), t)

    # Compose I = J*t + A*(1-t)
    I = np.empty_like(J)
    for c in range(3):
        I[...,c] = J[...,c]*t + (A[c])*(1.0 - t)

    # (Optional) add slight sensor bloom/glow in highlights for realism
    if random.random()<0.5:
        hsv = cv2.cvtColor((I*255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[...,2] = np.clip(hsv[...,2]*random.uniform(1.0,1.08), 0, 255)
        I = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)/255.0

    return (I*255.0).clip(0,255).astype(np.uint8)

def process_folder(img_dir, coco_json, out_dir, split="train"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(coco_json))
    # index anns by image id
    by_img = {}
    for a in data.get("annotations", []):
        by_img.setdefault(a["image_id"], []).append(a)
    for im in data["images"]:
        img_path = Path(img_dir)/im["file_name"]
        if not img_path.exists(): continue
        img = cv2.imread(str(img_path))
        anns = by_img.get(im["id"], [])
        hazy = synthesize_haze(img, anns=anns, keep_text_readable=True)
        out_path = out_dir/im["file_name"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), hazy)
    print("Done:", out_dir)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", required=True, help="COCO images root")
    parser.add_argument("--coco_json", required=True, help="COCO (or COCO-Text-like) annotations with bboxes")
    parser.add_argument("--out_dir", required=True, help="where to write hazy images")
    args = parser.parse_args()
    # global model load once
    _MIDAS = load_midas("DPT_Small")
    process_folder(args.img_dir, args.coco_json, args.out_dir)
