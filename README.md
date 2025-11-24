#  STR in Adverse Weather Conditions

Scene Text Recognition Pipeline for Rainy, Snowy, and Hazy Weather

A modular deep learning pipeline for **weather-aware scene text recognition (STR)**.

The system dynamically classifies weather using **CLIP**, detects text using **CRAFT**, and recognizes text using **PARSeq** models **fine-tuned per weather condition**.

---

## Repository Structure


```
advstr/
│
├── clip/ # Weather classifier
│
├── craft/
│ ├── model/ # Original CRAFT .pth weights
│ └── train/ # Unused for this project
│
├── parseq/
│ ├── checkpoints/
│ │ ├── parseq_rainy.ckpt
│ │ ├── parseq_snowy.ckpt
│ │ ├── parseq_hazy.ckpt
│ │ └── parseq_original.pt
│ ├── data
│ ├── train/{rainy, snowy, hazy}
│ ├── val/{rainy, snowy, hazy}
│ └── test/{rainy, snowy, hazy}
│
├── run.py # Full inference pipeline
└── advstr.yaml # Conda environment file


```

## Environment Setup

### Create environment

```bash
conda env create -f advstr.yaml
conda activate advstr
```

### Set PYTHONPATH

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```


### advstr.yaml

```yaml

name: advstr\
channels:\
  - defaults\
  - pytorch\
  - conda-forge

dependencies:\
  - python=3.10\
  - pip\
  - pytorch==2.5.1\
  - torchvision==0.20.1\
  - cudatoolkit=11.8\
  - numpy\
  - pillow\
  - opencv\
  - scikit-image\
  - shapely\
  - pyyaml\
  - pip:\
      - timm\
      - lmdb\
      - tqdm\
      - regex\
      - ftfy\
      - clip-anytorch\
      - pytorch-lightning==1.8.6
      - einops

```


###  Dataset Structure (LMDB)


```
parseq/data/\
├── train/{rainy | snowy | hazy}
├── val/{rainy | snowy | hazy}
└── test/{rainy | snowy | hazy}


```

Each folder contains LMDB datasets prepared from RoadText1K samples.
🧠 Fine-Tuning PARSeq\
Example (Rainy)

```bash

python3 parseq/train.py \\
    dataset=rainy \\
    pretrained=parseq/checkpoints/parseq_original.pt \\
    model.batch_size=32 \\
    trainer.devices=1 \\
    trainer.max_epochs=20 \\
    trainer.val_check_interval=60

```

Produces:

```
parseq_rainy.ckpt
parseq_snowy.ckpt
parseq_hazy.ckpt

```

---

## Testing

```bash

python3 parseq/test.py parseq/checkpoints/parseq_hazy.ckpt \\
    --data_root parseq/data/test/hazy \\
    --batch_size 32 \\
    --num_workers 4


```

---

## Results

Original PARSeq

|Dataset   | Accuracy  |  1-NED  |  Confidence  |  Label Length |
| -------- | --------- | ------- | ------------ | ------------- |
|Rainy     | 50.30     | 74.43   |   59.24      |  5.89         |
|Snowy     | 64.94     | 83.91   |   73.08      |   5.81        |
|Hazy      | 74.09     | 88.45   |   78.34      |   5.93        |

Fine-Tuned PARSeq

|Dataset   | Accuracy  |  1-NED  |  Confidence  |  Label Length |
| -------- | --------- | ------- | ------------ | ------------- |
|Rainy     | 58.92     | 79.43   |   63.80      |  5.90         |
|Snowy     | 70.20     | 87.57   |   73.24      |   5.98        |
|Hazy      | 77.21     | 90.29   |   77.07      |   5.99        |

---

## Inference (run.py)

Example


```bash

python3 run.py --image sample_images/clean.jpg


```

run.py performs:

```bash

    Weather recognition (CLIP)

    Weather-adaptive PARSeq selection

    CRAFT text detection → bounding boxes

    PARSeq recognition

    Output printed + saved

```

### Required paths
```bash

CRAFT_WEIGHTS = "craft/model/craft_mlt_25k.pth"

PARSEQ_WEIGHTS = {\
    "rainy": "parseq/checkpoints/parseq_rainy.ckpt",
    "snowy": "parseq/checkpoints/parseq_snowy.ckpt",
    "hazy":  "parseq/checkpoints/parseq_hazy.ckpt",
}

```

---

## Requirements


```bash

pip install timm lmdb tqdm regex ftfy clip-anytorch einops


```

(PyTorch is installed via Conda in advstr.yaml.)

## Pipeline Overview

```bash

Input Image
    ↓
CLIP Weather Classifier
    ↓
Select PARSeq Model (Rainy/Snowy/Hazy)
    ↓
CRAFT Text Detector
    ↓
PARSeq Text Recognition
    ↓
Final Output Text

```
