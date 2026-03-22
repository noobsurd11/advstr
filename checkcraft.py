import torch
import cv2
import numpy as np

from craft.craft import CRAFT
import craft.imgproc as imgproc
import craft.craft_utils as craft_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_PATH = "sample_imgs/710-0000297.jpg"   # change if needed

def load_craft(weight_path):
    print(f"[INFO] Loading CRAFT from {weight_path}")
    net = CRAFT(pretrained=False).to(DEVICE)

    state = torch.load(weight_path, map_location=DEVICE)
    new_state = {}
    for k, v in state.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_k] = v

    missing, unexpected = net.load_state_dict(new_state, strict=False)
    print("[DEBUG] missing keys:", missing)
    print("[DEBUG] unexpected keys:", unexpected)

    net.eval()
    return net

def forward_and_stats(net, name):
    img = cv2.imread(IMG_PATH)
    img_resized, ratio, _ = imgproc.resize_aspect_ratio(
        img, 1280, interpolation=cv2.INTER_LINEAR, mag_ratio=1.5
    )
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    print(f"\n==== {name} ====")
    print("text  min/mean/max:", score_text.min(), score_text.mean(), score_text.max())
    print("link  min/mean/max:", score_link.min(), score_link.mean(), score_link.max())

    # Try a low threshold detection
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link,
        text_threshold=0.4,
        link_threshold=0.2,
        low_text=0.2,
        poly=False,
    )
    print(f"{name}: #boxes with low thresholds = {len(boxes)}")

def main():
    base_weight = "craft/model/craft_mlt_25k.pth"
    ft_weight   = "craft/model/CRAFT_rainy_finetuned.pth"

    net_base = load_craft(base_weight)
    net_ft   = load_craft(ft_weight)

    forward_and_stats(net_base, "BASE (mlt_25k)")
    forward_and_stats(net_ft,   "FINETUNED (rainy)")

if __name__ == "__main__":
    main()
