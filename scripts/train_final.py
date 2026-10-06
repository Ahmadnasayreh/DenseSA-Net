"""Train DenseSA-Net, VGG16 and CustomCNN on the fixed split of each dataset.

Weights are written to <out>/<dataset>/<model>.pt and are used by
evaluate.py and explainability.py.

    python scripts/train_final.py --data-root data --out outputs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from densesa.config import DATASETS, MODELS, add_common_args, config_from_args
from densesa.data import load_dataset, make_loader, split_dataset
from densesa.engine import fit, get_device, set_seed
from densesa.models import build_model


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = get_device()

    for name in args.datasets:
        classes = DATASETS[name]["classes"]
        train, val, test, _ = split_dataset(load_dataset(args.data_root, name), cfg.seed)
        print(f"{name}: train {len(train)}  val {len(val)}  test {len(test)}")
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        train_loader = make_loader(train, classes, cfg, augment=True)
        val_loader = make_loader(val, classes, cfg)

        for model_name in args.models:
            set_seed(cfg.seed)
            print(f"training {model_name}")
            model = build_model(model_name, len(classes), cfg.pretrained).to(device)
            history, seconds = fit(model, train_loader, val_loader, cfg, device)
            torch.save(model.state_dict(), out / f"{model_name}.pt")
            history.to_csv(out / f"history_{model_name}.csv", index=False)
            print(f"  finished in {seconds / 60:.1f} min")


if __name__ == "__main__":
    main()
