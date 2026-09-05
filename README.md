# Basketball Score Tracking

A computer vision project for detecting basketball game elements and building toward automatic shot tracking and live score estimation from video.

## Project Goal

The long-term pipeline is:

```
Phone Camera / Recorded Video
        ↓
Object Detection
        ↓
Object Tracking
        ↓
Shot Detection
        ↓
Made / Missed
        ↓
Score Logic
        ↓
Live Scoreboard
```

The current stage focuses on building and training an object detector for:

- `basketball`
- `hoop`
- `player`

The phone is intended mainly as the video source, while inference can initially run on a laptop GPU.

## Dataset Sources

The final training dataset was created by combining three basketball object-detection datasets from Roboflow, exported in YOLO11 format:

- `Basketball detection.v1i.yolov11`
- `Basketball Game Detections.v9i.yolov11`
- `Basketball.v1i.yolov11`

The original datasets used different class definitions, so their annotations were remapped into the three unified classes (`basketball`, `hoop`, `player`). The original train/validation/test splits were not preserved for training — all samples were merged into a single pool, cleaned, and then divided into a new leakage-aware split.

## Dataset Preparation

Three basketball object-detection datasets were merged into one unified pool. The original source `train` / `valid` / `test` splits were preserved only as metadata — a completely new split was created after cleaning.

### Unified Classes

All source labels were mapped into:

| ID | Class |
|----|-------|
| 0 | basketball |
| 1 | hoop |
| 2 | player |

Annotations such as `FG Attempt`, `FG Made`, `Ball in Basket`, and referee labels were excluded, since shot outcomes will later be determined using tracking and temporal logic rather than static labels.

### Data Cleaning Pipeline

1. **Dataset structure inspection** — verified image files, label files, and YOLO annotation format (`class_id x_center y_center width height`, normalized coordinates).
2. **Class mapping** — converted labels from all three source datasets to the unified three-class format.
3. **Dataset merge** — merged all source images into one temporary pool, preserving source dataset, original split, original filename, and merged filename in `source_manifest.csv`.
4. **Missing file validation** — checked images/labels for consistency.
   - Missing clean labels: `0`
   - Removal entries missing from dataset: `0`
5. **Class distribution analysis** — final planned split:

   | Class | Train | Validation | Test |
   |-------|------:|-----------:|-----:|
   | basketball | 28,827 | 3,570 | 3,553 |
   | hoop | 19,908 | 2,494 | 2,602 |
   | player | 62,975 | 7,464 | 7,988 |

   Classes were not forced to have equal annotation counts, since basketball scenes naturally contain more players than hoops or balls.

6. **Annotation visualization** — random samples visualized with ground-truth boxes to verify class correctness, box placement, annotation quality, and dataset relevance.
7. **Bounding-box QA**
   - Maximum annotations per image: `20`
   - Tiny-box review thresholds: basketball ≤ 6px, hoop ≤ 12px, player ≤ 8px
   - Same-class boxes with IoU ≥ 0.98 flagged as possible duplicates
   - Large boxes were analyzed but not used as a rejection criterion, since valid close-up images can naturally contain large objects
8. **Manual QA** — suspicious samples exported for review (incorrect classes, inaccurate boxes, missing annotations, duplicates, unusable images, low gameplay relevance). Rejected samples tracked in `removal_list.csv`. The original merged dataset was never modified during QA.
9. **Duplicate analysis**
   - *Exact duplicates* (SHA-256): 411 duplicate groups, 825 images inside duplicate groups, 414 redundant exact copies. Groups with identical labels were handled automatically; groups with differing labels were manually compared and the best-labeled version kept.
   - *Near duplicates* (perceptual hashing): many visually similar images found, as expected given repeated courts, camera angles, formations, and consecutive frames — these were kept rather than removed.
10. **Blur analysis** — sharpness measured via variance of the Laplacian.
    - Blur threshold: `46.52`
    - Images reviewed: `182`
    - Images rejected: `76`
    - Blurred images were not removed automatically, since motion blur is natural in basketball footage.

### Final Clean Pool

| | Count |
|---|---:|
| Original images | 36,977 |
| Removed samples | 782 |
| **Clean samples** | **36,195** |

### Leakage-Aware Split

A normal random split could place multiple Roboflow variants of the same original image into different subsets. To avoid this, samples were grouped by `source_dataset + original filename before ".rf."`, and all variants of the same original image were assigned to the same split.

**Group statistics:**

| | Count |
|---|---:|
| Clean samples | 36,195 |
| Unique groups | 11,426 |
| Single-image groups | 3,187 |
| Multi-image groups | 8,239 |
| Largest group | 24 |

**Final split** — reproducible group-aware 80 / 10 / 10:

| Split | Images |
|-------|-------:|
| Train | 28,956 |
| Validation | 3,620 |
| Test | 3,619 |
| **Total** | **36,195** |

**Integrity checks:**

- Train / Validation overlap: `0`
- Train / Test overlap: `0`
- Validation / Test overlap: `0`
- Unassigned samples: `0`

## Final Dataset Structure

```
data/
├── training/
│   ├── dataset_01/
│   ├── dataset_02/
│   └── dataset_03/
├── merged/
│   └── basketball_dataset_temp/
└── processed/
    └── basketball_dataset_final/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── valid/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        ├── data.yaml
        └── final_manifest.csv
```

`final_manifest.csv` preserves the connection between every final image and its original source dataset and filename.

## Notebooks

- **`01_data_preparation.ipynb`** — the complete dataset preparation workflow: source inspection, class mapping, dataset merge, QA, manual review, duplicate analysis, blur analysis, clean-pool creation, group-aware split, and final dataset creation.

The next notebook will focus on model training and evaluation.

## Model Direction

The first main detector planned for fine-tuning is **YOLO26s**.

Evaluation will focus on both overall and per-class metrics:

- Precision
- Recall
- mAP50
- mAP50-95

Special attention will be given to **basketball recall**, since missed ball detections can strongly affect later tracking and shot-detection stages.

## Next Steps

```
Load final dataset
        ↓
Visual sanity check
        ↓
Train YOLO26s
        ↓
Evaluate validation metrics
        ↓
Per-class analysis
        ↓
Error analysis
        ↓
Improve detector
        ↓
Object tracking
        ↓
Shot detection
        ↓
Made / missed logic
        ↓
Live scoreboard
```

## Tech Stack

- Python
- Jupyter Notebook
- OpenCV
- NumPy
- PyTorch
- Ultralytics YOLO
- CUDA / NVIDIA GPU
