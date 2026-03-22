# -*- coding: utf-8 -*-

import argparse
import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from config.load_config import load_yaml, DotDict
from model.craft import CRAFT
from metrics.eval_det_iou import DetectionIoUEvaluator
from utils.inference_boxes import test_net
from utils.util import copyStateDict

#######################################################################
# CUSTOM DATASET LOADER (your exact ICDAR format)
#######################################################################

def load_custom_gt(root_dir):
    """
    root_dir/
        ch4_test_images/
        ch4_test_localization_transcription_gt/
    """

    img_dir = os.path.join(root_dir, "ch4_test_images")
    gt_dir = os.path.join(root_dir, "ch4_test_localization_transcription_gt")

    img_paths = sorted([
        os.path.join(img_dir, x) for x in os.listdir(img_dir)
        if x.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    total_gt = []

    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        gt_name = f"gt_{img_name.split('.')[0]}.txt"
        gt_path = os.path.join(gt_dir, gt_name)

        boxes = []

        if not os.path.exists(gt_path):
            print(f"[WARNING] GT file not found: {gt_path}")
            total_gt.append([])
            continue

        with open(gt_path, "r", encoding="utf-8-sig") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(',')
            coords = list(map(int, parts[:8]))
            text = parts[8] if len(parts) > 8 else "###"

            poly = np.array(coords).reshape(-1, 2)

            boxes.append({
                "points": poly,
                "text": text,
                "ignore": text == "###"
            })

        total_gt.append(boxes)

    return total_gt, img_paths


#######################################################################
# VISUALIZATION HELPERS
#######################################################################

def save_viz(image, polys, gt_boxes, score_text, out_path):
    img = image.copy()

    # Draw predicted polys (green)
    for poly in polys:
        poly = np.array(poly).reshape(-1, 2).astype(int)
        cv2.polylines(img, [poly], True, (0, 255, 0), 2)

    # Draw GT polys (red)
    if gt_boxes is not None:
        for gt in gt_boxes:
            gpoly = gt["points"].astype(int)
            cv2.polylines(img, [gpoly], True, (0, 0, 255), 2)

    cv2.imwrite(out_path, img)


#######################################################################
# MAIN EVAL FUNCTION
#######################################################################

def main_eval(model_path, config):

    evaluator = DetectionIoUEvaluator()
    gt_boxes, img_paths = load_custom_gt(config.test_data_dir)

    print(f"[INFO] Loaded {len(img_paths)} images")

    # Load CRAFT
    model = CRAFT()

    print(f"[INFO] Loading model from {model_path}")
    net = torch.load(model_path, map_location="cpu")
    if "craft" in net:
        model.load_state_dict(copyStateDict(net["craft"]))
    else:
        model.load_state_dict(copyStateDict(net))

    if config.cuda:
        model = model.cuda()

    model.eval()

    total_preds = []

    os.makedirs(config.result_dir, exist_ok=True)

    for idx, img_path in enumerate(tqdm(img_paths)):

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bboxes, polys, score_text = test_net(
            model,
            image,
            config.text_threshold,
            config.link_threshold,
            config.low_text,
            config.cuda,
            config.poly,
            config.canvas_size,
            config.mag_ratio,
        )

        # Convert preds to ICDAR format
        pred_boxes = []
        for poly in bboxes:
            pred_boxes.append({
                "points": np.array(poly),
                "text": "###",
                "ignore": False
            })

        total_preds.append(pred_boxes)

        # Visualization
        if config.vis_opt:
            name = os.path.basename(img_path)
            out_image_path = os.path.join(config.result_dir, f"viz_{name}")
            save_viz(image[:, :, ::-1], polys, gt_boxes[idx], score_text, out_image_path)

    # Compute IoU metrics
    results = []
    for gt, pred in zip(gt_boxes, total_preds):
        per_img = evaluator.evaluate_image(gt, pred)
        results.append(per_img)

    metrics = evaluator.combine_results(results)
    print("\n==================== RESULTS ====================")
    print(metrics)
    print("=================================================")

    return metrics


#######################################################################
# SCRIPT ENTRY POINT
#######################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml", default="custom_data_train", type=str)
    args = parser.parse_args()

    cfg = DotDict(load_yaml(args.yaml))

    # Set output directory
    cfg["result_dir"] = os.path.join("exp", args.yaml, "custom_eval")

    main_eval(cfg.test.trained_model, cfg)
