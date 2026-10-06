"""Test-set evaluation of the trained models (Tables 9-12, Figures 10-11,
parameter counts of Table 5).

    python scripts/evaluate.py --data-root data --out outputs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from densesa import plotting
from densesa.config import DATASETS, MODELS, add_common_args, config_from_args
from densesa.data import load_dataset, make_loader, split_dataset
from densesa.engine import compute_metrics, get_device, per_class_table, predict
from densesa.models import build_model, count_parameters


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--warmup", type=int, default=5,
                        help="batches discarded before timing inference")
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    results, perclass, params = [], [], []

    for name in args.datasets:
        classes = DATASETS[name]["classes"]
        _, _, test, _ = split_dataset(load_dataset(args.data_root, name), cfg.seed)
        loader = make_loader(test, classes, cfg)
        folder = Path(args.out) / name
        cms, accs = {}, {}

        for model_name in args.models:
            weights = folder / f"{model_name}.pt"
            if not weights.exists():
                print(f"skipping {name}/{model_name}: {weights} not found")
                continue
            model = build_model(model_name, len(classes), pretrained=False)
            model.load_state_dict(torch.load(weights, map_location="cpu"))
            model.to(device)
            labels, preds, probs, ms = predict(model, loader, device, cfg, args.warmup)
            m = compute_metrics(labels, preds, probs, len(classes))
            total, stage1 = count_parameters(model)
            cms[model_name], accs[model_name] = m.pop("cm"), m["Accuracy"]
            results.append({"Dataset": name, "Model": model_name, **m,
                            "Params_M": total / 1e6, "ms_per_image": ms})
            params.append({"Dataset": name, "Model": model_name,
                           "Total": total, "Trained_in_stage1": stage1})
            table = per_class_table(labels, preds, classes)
            table.insert(0, "Model", model_name)
            table.insert(0, "Dataset", name)
            perclass.append(table)
            print(f"{name:9s} {model_name:11s} acc {m['Accuracy']:.4f}  f1 {m['F1-Score']:.4f}"
                  f"  auc {m['AUC']:.4f}  {total / 1e6:.2f}M  {ms:.2f} ms/img")
            if model_name == "DenseSANet":
                plotting.per_class_bars(table, f"Per-class metrics, DenseSA-Net, {name}",
                                        folder / "perclass.png", args.dpi)
            del model
            torch.cuda.empty_cache()

        if cms:
            plotting.confusion_grid(cms, accs, classes, f"Confusion matrices, {name}",
                                    folder / "confusion.png", args.dpi)

    out = Path(args.out)
    pd.DataFrame(results).to_csv(out / "test_results.csv", index=False)
    pd.concat(perclass).to_csv(out / "perclass.csv", index=False)
    pd.DataFrame(params).to_csv(out / "parameters.csv", index=False)


if __name__ == "__main__":
    main()
