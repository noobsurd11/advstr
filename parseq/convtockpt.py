import torch

# ---- Paths ----
PT_PATH      = "parseq/checkpoints/parseq.pt"                # raw pretrained weights
REF_CKPT     = "parseq/checkpoints/parseq_rainy.ckpt"         # any *working* Lightning ckpt (hazy/rainy/snowy)
OUT_CKPT     = "parseq/checkpoints/parseq0.ckpt" # output ckpt to use in test.py

print(f"Loading raw .pt weights from: {PT_PATH}")
raw_sd = torch.load(PT_PATH, map_location="cpu")
if "state_dict" in raw_sd:
    # Just in case, but for you it should be a plain OrderedDict
    raw_sd = raw_sd["state_dict"]
print(f"Raw keys (no prefix) example: {list(raw_sd.keys())[:5]}")

print(f"Loading reference Lightning checkpoint from: {REF_CKPT}")
ref_ckpt = torch.load(REF_CKPT, map_location="cpu")

if "state_dict" not in ref_ckpt:
    raise RuntimeError("Reference ckpt has no 'state_dict' key, something is wrong.")

ref_state = ref_ckpt["state_dict"].copy()
print(f"Example keys in reference state_dict: {list(ref_state.keys())[:5]}")

# ---- Replace model.* weights with those from raw .pt ----
num_replaced = 0
num_missing  = 0
for k, v in raw_sd.items():
    prefixed = "model." + k
    if prefixed in ref_state:
        ref_state[prefixed] = v
        num_replaced += 1
    else:
        # Not fatal, but useful to know
        print(f"[WARN] Raw key '{k}' (-> '{prefixed}') not found in reference state_dict")
        num_missing += 1

print(f"Replaced {num_replaced} parameters from raw .pt into reference ckpt.")
if num_missing:
    print(f"{num_missing} raw keys did not match any 'model.*' key in reference.")

# ---- Build new Lightning-style checkpoint ----
new_ckpt = {
    "state_dict": ref_state,
    "hyper_parameters": ref_ckpt.get("hyper_parameters", {}),
    "pytorch-lightning_version": ref_ckpt.get("pytorch-lightning_version", "2.0.0"),
}

torch.save(new_ckpt, OUT_CKPT)
print("Saved new Lightning checkpoint:", OUT_CKPT)
