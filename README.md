# DenseSA-Net

Code for the paper *DenseSA-Net: A Parameter-Efficient Spatial-Attention Architecture
for Explainable Multi-Class Blood Cell Classification* (Scientific Reports, under review).

DenseSA-Net is a DenseNet-121 backbone followed by a single 7x7 spatial-attention
module (50 parameters). The backbone features and the attention-weighted features
are concatenated before global average pooling, so the classifier always keeps an
unmodified path from the backbone. The model has 6.97 M parameters.

The repository also contains everything else reported in the paper: the full 2^3
factorial ablation that led to this design (dilated convolutions, channel attention,
spatial attention), the two baselines (VGG16 and a small CNN), the repeated
cross-validation with its statistical tests, and the quantitative check of the
attention maps against nuclear regions.

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
```

All experiments were run on one NVIDIA GeForce RTX 5070 (12 GB) under Windows.
`num_workers` defaults to 0 for that reason; on Linux you can raise it with `--workers`.

## Data

The three datasets are public:

| Folder | Dataset | Classes | Images |
|--------|---------|---------|--------|
| Data1 | [Peripheral blood cells](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset) (Acevedo et al., 2020) | 8 | 17,092 |
| Data2 | [Blood cell images for cancer detection](https://www.kaggle.com/datasets/sumithsingh/blood-cell-images-for-cancer-detection) | 5 | 5,000 |
| Data3 | [Blood cell cancer ALL, 4 classes](https://www.kaggle.com/datasets/mohammadamireshraghi/blood-cell-cancer-all-4class) | 4 | 3,242 |

Put them under one folder, one sub-folder per class:

```
data/
  Data1/neutrophil/*.jpg, Data1/eosinophil/*.jpg, ...
  Data2/basophil/*.jpg, ...
  Data3/Benign/*.jpg, Data3/[Malignant] Pro-B/*.jpg, ...
```

An extra wrapper folder (for example `Data1/bloodcells_dataset/`) is detected automatically.
Each dataset is split once with seed 42: 10% is held out as the test set before
anything else, and the rest is split 80/10 into training and validation.

## Reproducing the results

| Script | What it produces | Paper |
|--------|------------------|-------|
| `scripts/train_final.py` | final DenseSA-Net, VGG16 and CustomCNN models | |
| `scripts/evaluate.py` | test metrics, per-class tables, confusion matrices, parameter counts | Tables 5, 9-12; Figures 10-11 |
| `scripts/factorial_ablation.py` | the eight ablation variants, main effects and interactions | Tables 7-8; Figures 8-9 |
| `scripts/repeated_cv.py` | 3 x 7-fold cross-validation, Friedman and Wilcoxon tests, effect sizes, bootstrap CIs, hold-out evaluation of the fold models | Tables 13-15; Figure 12 |
| `scripts/explainability.py` | Grad-CAM and attention maps, failure cases, alignment with nuclear regions against chance | Table 16; Figures 13-15 |
| `scripts/merge_panels.py` | combines the per-dataset panels into the published multi-panel figures | |
| `scripts/summarise_results.py` | recomputes the statistics from the files in `results/` | Tables 13, 14, 16 |

```
python scripts/train_final.py        --data-root data --out outputs
python scripts/evaluate.py           --data-root data --out outputs
python scripts/explainability.py     --data-root data --out outputs
python scripts/factorial_ablation.py --data-root data --out outputs
python scripts/repeated_cv.py        --data-root data --out outputs
```

`--datasets Dataset1` restricts a run to one dataset. Rough run times on the RTX 5070:
the factorial ablation takes about 5 hours for the three datasets and the repeated
cross-validation about a day and a half. `repeated_cv.py` saves its progress after
every fold and resumes if it is started again with the same arguments.

## Checking the statistics without retraining

`results/` contains the fold accuracies of the repeated cross-validation for the three
models and the per-image alignment measures of the Grad-CAM analysis, as obtained in
the paper. The tables can be recomputed from them in a few seconds:

```
python scripts/summarise_results.py
```

## Training protocol

| | |
|---|---|
| Stage 1 | backbone frozen; attention module and head trained, lr 1e-3, 5 epochs |
| Stage 2 | all layers trained, lr 1e-4, up to 20 epochs |
| Optimiser | AdamW, weight decay 1e-4 |
| Schedule | learning rate halved every 3 epochs within each stage |
| Early stopping | patience 5 on validation loss; best weights restored |
| Batch size | 64, gradient clipping at 1.0, mixed precision on GPU |
| Augmentation | horizontal flip, rotation within +/-10 degrees (training only) |

VGG16 and CustomCNN have no separately added module and are trained end to end in
both stages with the same learning rates. No hyperparameter search was run; every
model uses the settings above.

## Using a trained model

```python
import torch
from densesa.models import DenseSANet

model = DenseSANet(num_classes=8, pretrained=False)
model.load_state_dict(torch.load("outputs/Dataset1/DenseSANet.pt", map_location="cpu"))
model.eval()
logits = model(images)               # images: (B, 3, 224, 224), ImageNet-normalised
attention = model.last_attention     # (B, 1, 7, 7) spatial attention map
```

Variant F4 of the ablation has the same architecture as DenseSA-Net, so its weights
load into `DenseSANet` directly.

## Reproducibility

Seeds are fixed for the data split, the cross-validation partitions and model
initialisation. cuDNN kernels are not fully deterministic, so a rerun can differ from
the reported numbers in the last decimal places. Inference times depend on the GPU.

## Citation

```bibtex
@article{alnasra2026densesanet,
  title   = {DenseSA-Net: A Parameter-Efficient Spatial-Attention Architecture for
             Explainable Multi-Class Blood Cell Classification},
  author  = {AL-Nasra, Nazha Ala'a and Wedyan, Mohammad and Alturki, Ryan and
             Alkhateeb, Hamza K. and Alshawi, Bandar and Bashkami, Ayah and
             AL-Nasra, Dana Ala'a and Gharaibeh, Hasan and Nasayreh, Ahmad},
  journal = {Scientific Reports},
  year    = {2026},
  note    = {under review}
}
```

## Author

Code written and maintained by Ahmad Nasayreh.

## License

MIT, see `LICENSE`.
