"""Settings shared by all experiment scripts.

The class order below matters: it fixes the label indices, so trained
weights are only valid with the same ordering.
"""
from dataclasses import dataclass

DATASETS = {
    "Dataset1": {
        "folder": "Data1",
        "classes": ["neutrophil", "eosinophil", "ig", "platelet",
                    "erythroblast", "monocyte", "basophil", "lymphocyte"],
    },
    "Dataset2": {
        "folder": "Data2",
        "classes": ["basophil", "erythroblast", "monocyte", "myeloblast",
                    "seg_neutrophil"],
    },
    "Dataset3": {
        "folder": "Data3",
        "classes": ["[Malignant] Pro-B", "[Malignant] Pre-B",
                    "[Malignant] early Pre-B", "Benign"],
    },
}

MODELS = ["DenseSANet", "VGG16", "CustomCNN"]

# Repeated cross-validation: one stratified 7-fold split per seed.
N_FOLDS = 7
CV_SEEDS = (42, 2024, 7)


@dataclass
class TrainConfig:
    img_size: int = 224
    batch_size: int = 64
    lr_stage1: float = 1e-3
    lr_stage2: float = 1e-4
    epochs_stage1: int = 5
    epochs_stage2: int = 20
    weight_decay: float = 1e-4
    patience: int = 5          # early stopping on validation loss
    grad_clip: float = 1.0
    lr_step: int = 3           # StepLR: halve the learning rate every 3 epochs
    lr_gamma: float = 0.5
    num_workers: int = 0       # keep at 0 on Windows
    amp: bool = True
    seed: int = 42
    pretrained: bool = True    # ImageNet weights for DenseNet-121 / VGG16


def add_common_args(parser):
    """Arguments used by every script."""
    parser.add_argument("--data-root", required=True,
                        help="folder that contains Data1, Data2 and Data3")
    parser.add_argument("--out", default="outputs", help="output folder")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS),
                        choices=list(DATASETS))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--epochs-stage1", type=int, default=5)
    parser.add_argument("--epochs-stage2", type=int, default=20)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="random initialisation (only useful for quick tests)")
    parser.add_argument("--dpi", type=int, default=600)
    return parser


def config_from_args(args):
    return TrainConfig(img_size=args.img_size, batch_size=args.batch_size,
                       epochs_stage1=args.epochs_stage1,
                       epochs_stage2=args.epochs_stage2,
                       num_workers=args.workers, amp=not args.no_amp,
                       pretrained=not args.no_pretrained)
