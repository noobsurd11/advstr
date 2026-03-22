#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import torch
import numpy as np
import cv2
from PIL import Image

# ====== IMPORT CRAFT (NAVER) ======
from craft.craft import CRAFT
import craft.craft_utils as craft_utils
import craft.imgproc as imgproc

# ====== PARSEQ IMPORTS ======
from parseq.strhub.models.utils import load_from_checkpoint, parse_model_args
from parseq.strhub.data.module import SceneTextDataModule
from parseq.strhub.models.parseq.system import PARSeq


import clip
import torch.nn.functional as F

# ====== WEATHER → WEIGHT PATHS ======
WEATHER_WEIGHTS = {
    "rainy": {
        "craft": "craft/model/craft.pth",
        "parseq": "parseq/checkpoints/parseq_rainy.ckpt"
    },
    "hazy": {
        "craft": "craft/model/craft.pth",
        "parseq": "parseq/checkpoints/parseq_hazy.ckpt"
    },
    "snowy": {
        "craft": "craft/model/craft.pth",
        "parseq": "parseq/checkpoints/parseq_snowy.ckpt"
    },
}


# ----------------------------------------------------------
#               CLIP CLASSIFICATION
# ----------------------------------------------------------

def classify_weather_with_clip(img_bgr, device):
    # Load CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Convert BGR → RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    # CLIP preprocessing
    clip_input = preprocess(pil).unsqueeze(0).to(device)

    # Three textual prompts
    classes = ["rainy weather", "foggy or hazy weather", "snowy weather"]
    text_tokens = clip.tokenize(classes).to(device)

    # Forward
    with torch.no_grad():
        img_feat = model.encode_image(clip_input)
        txt_feat = model.encode_text(text_tokens)

        img_feat = F.normalize(img_feat, dim=-1)
        txt_feat = F.normalize(txt_feat, dim=-1)

        logits = (img_feat @ txt_feat.T)[0]
        idx = logits.argmax().item()

    return classes[idx]
# ----------------------------------------------------------
#               LOAD CRAFT MODEL
# ----------------------------------------------------------
def load_craft(weight_path, device):
    print(f"[INFO] Loading CRAFT weights: {weight_path}")
    net = CRAFT(pretrained=False).to(device)

    # Load checkpoint with key-fix logic (like test.py)
    state = torch.load(weight_path, map_location=device)
    new_state = {}

    # remove "module." if present
    for k, v in state.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        new_state[new_k] = v

    net.load_state_dict(new_state, strict=False)
    net.eval()
    return net


# ----------------------------------------------------------
#               CRAFT DETECTION
# ----------------------------------------------------------
def craft_detect(net, image, device,
                 text_threshold=0.7,
                 link_threshold=0.4,
                 low_text=0.4,
                 canvas_size=1280,
                 mag_ratio=1.5,
                 poly=False):

    # resize
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio
    )
    ratio_h = ratio_w = 1 / target_ratio

    # normalize
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().numpy()
    score_link = y[0, :, :, 1].cpu().numpy()

    # detect boxes
    boxes, polys = craft_utils.getDetBoxes(
        score_text, score_link, text_threshold, link_threshold, low_text, poly
    )

    # scale coords back to original image
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return boxes


# ----------------------------------------------------------
#               LOAD PARSEQ MODEL
# ----------------------------------------------------------
# def load_parseq(path, device, extra_args):
#     print(f"[INFO] Loading PARSEQ checkpoint: {path}")
#     model = load_from_checkpoint(path, **extra_args).eval().to(device)
#     transform = SceneTextDataModule.get_transform(model.hparams.img_size)
#     return model, transform

# def load_parseq(path, device, extra_args):
#     print(f"[INFO] Loading PARSEQ weights (torch state_dict): {path}")

#     # 1. Load base model architecture (same used in torch.hub)
#     model = torch.hub.load('baudm/parseq', 'parseq', pretrained=False)

#     # 2. Load your downloaded state_dict
#     state = torch.load(path, map_location=device)
#     if "state_dict" in state:
#         # handle case where Lightning-style dict is wrapped
#         state = state["state_dict"]

#     model.load_state_dict(state, strict=True)
#     model = model.to(device).eval()

#     # 3. Build transforms
#     from parseq.strhub.data.module import SceneTextDataModule
#     transform = SceneTextDataModule.get_transform(model.hparams.img_size)

#     return model, transform

# 
def load_parseq(path, device, extra_args):
    print(f"[INFO] Loading PARSEQ local .pt: {path}")

    # 1. load pretrained model architecture
    model = torch.hub.load('baudm/parseq', 'parseq', pretrained=True)  
    # pretrained=True only loads architecture; we will overwrite weights next.

    # 2. load the actual .pt weights
    state = torch.load(path, map_location=device)

    # 3. strict=False is REQUIRED because Lightning names differ
    model.load_state_dict(state, strict=False)

    model = model.eval().to(device)
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    return model, transform
   

#def load_parseq(path, device, extra_args=None):
   # print(f"[INFO] Loading PARSEQ checkpoint (manual class load): {path}")

    # 1. Load checkpoint dict
   # ckpt = torch.load(path, map_location=device)

    # 2. Extract hyperparameters used in training
   # hparams = ckpt["hyper_parameters"]

    # 3. Build model using same hyperparameters
   # model = PARSeq(**hparams).to(device)

    # 4. Load trained weights
  #  state_dict = ckpt["state_dict"]
  #  model.load_state_dict(state_dict, strict=True)

  #  model.eval()

    # 5. Build transform from img_size
    #transform = SceneTextDataModule.get_transform(hparams["img_size"])
 #   return model, transform




# ----------------------------------------------------------
#               PARSEQ RECOGNITION
# ----------------------------------------------------------
def recognize(model, transform, crop, device):
    pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        prob = model(tensor).softmax(-1)
        pred, _ = model.tokenizer.decode(prob)
    return pred[0].strip()


# ----------------------------------------------------------
#               FULL PIPELINE
# ----------------------------------------------------------
def run_pipeline(image_path, device, extra_args):
    # load image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    # -------- WEATHER CLASSIFICATION USING CLIP --------
    # -------- WEATHER CLASSIFICATION USING CLIP --------
    weather = classify_weather_with_clip(img, device)
    print(f"[INFO] Detected weather: {weather}")

    # map weather → internal key
    if "rain" in weather:
        key = "rainy"
    elif "fog" in weather or "hazy" in weather:
        key = "hazy"
    elif "snow" in weather:
        key = "snowy"
    else:
        key = "hazy"   # default (fog/haze is safest for text detection)

    # override paths (STOP using CLI args)
    craft_weight = WEATHER_WEIGHTS[key]["craft"]
    parseq_weight = WEATHER_WEIGHTS[key]["parseq"]

    print(f"[INFO] Selected CRAFT weight: {craft_weight}")
    print(f"[INFO] Selected PARSEQ weight: {parseq_weight}")


    # load models
    craft = load_craft(craft_weight, device)
    parseq, transform = load_parseq(parseq_weight, device, extra_args)

    # detect
    print("[INFO] Detecting text regions...")
    boxes = craft_detect(craft, img, device)
    print(f"[INFO] Found {len(boxes)} regions")

    results = []
    for box in boxes:
        box = np.array(box).astype(int)

        x_min = min(box[:, 0])
        x_max = max(box[:, 0])
        y_min = min(box[:, 1])
        y_max = max(box[:, 1])

        crop = img[y_min:y_max, x_min:x_max]
        if crop.size == 0:
            continue

        text = recognize(parseq, transform, crop, device)
        results.append({"text": text, "box": box.tolist()})

    return results


# ----------------------------------------------------------
#                CLI ENTRY
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda")
    args, unknown = parser.parse_known_args()

    extra_args = parse_model_args(unknown)

    output = run_pipeline(
        image_path=args.image,
        device=args.device,
        extra_args=extra_args
    )

    print("\n========== RESULTS ==========")
    for r in output:
        print(r["text"], " :: ", r["box"])
