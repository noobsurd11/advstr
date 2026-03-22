import torch
import cv2
import numpy as np
import argparse
from torchvision import transforms

# ------------------------------------------------------------
# Load MiDaS model (monocular depth, works for ANY image)
# ------------------------------------------------------------
def load_midas():
    midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")  # best balance
    midas.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    midas.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.dpt_transform
    return midas, transform, device


def estimate_depth(midas, transform, device, img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)
        if isinstance(prediction, dict):
            prediction = prediction["pred"]

        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)

    # Invert depth:
    depth = 1 - depth
    return depth


# ------------------------------------------------------------
# Atmospheric light estimation
# ------------------------------------------------------------
def get_atmospheric_light(img, t):
    h, w, _ = img.shape
    flat_t = t.reshape(-1)
    num = int(0.001 * len(flat_t))
    if num < 1: num = 1

    indices = np.argpartition(flat_t, num)[:num]
    ys = indices // w
    xs = indices % w

    brightest = 0
    A = None
    for i in range(len(xs)):
        intensity = img[ys[i], xs[i]].sum()
        if intensity > brightest:
            brightest = intensity
            A = img[ys[i], xs[i]]

    return A.astype(np.float32) / 255.0


# ------------------------------------------------------------
# Main fog synthesis
# ------------------------------------------------------------
def add_fog(img, depth, beta=0.02):
    # Smooth depth
    depth = cv2.GaussianBlur(depth, (21, 21), 15)

    # Physical transmission
    t = np.exp(-beta * depth).astype(np.float32)

    # UNIVERSAL GUIDED FILTER (fallback with bilateral)
    try:
        guided = cv2.ximgproc.guidedFilter(
            guide=img.astype(np.float32)/255.0,
            src=t,
            radius=20,
            eps=1e-3
        )
    except:
        # ALWAYS AVAILABLE - still gives good result
        guided = cv2.bilateralFilter(
            t, d=15, sigmaColor=0.1, sigmaSpace=15
        )

    guided = np.clip(guided, 0.0, 1.0)

    # Atmospheric light
    A = get_atmospheric_light(img, guided)

    J = img.astype(np.float32) / 255.0
    T = np.repeat(guided[:, :, None], 3, axis=2)

    I = J * T + A * (1 - T)
    return (I * 255).astype(np.uint8)



# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--out", default="fog.png")
    args = parser.parse_args()

    img = cv2.imread(args.img)
    if img is None:
        raise FileNotFoundError("Image not found.")

    midas, transform, device = load_midas()

    print("Estimating depth...")
    depth = estimate_depth(midas, transform, device, img)

    print("Adding fog...")
    fog = add_fog(img, depth, beta=args.beta)

    cv2.imwrite(args.out, fog)
    print("Saved:", args.out)
