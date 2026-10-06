"""Full 2^3 factorial ablation over dilated convolutions (D), channel
attention (C) and spatial attention (S): Tables 7-8, Figures 8-9.

All eight variants share the same split, seed and training protocol.
Variant F4 (spatial attention only) is DenseSA-Net; its saved weights can
be loaded directly into densesa.models.DenseSANet.

    python scripts/factorial_ablation.py --data-root data --out outputs
"""
import argparse
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from densesa import plotting
from densesa.config import DATASETS, add_common_args, config_from_args
from densesa.data import load_dataset, make_loader, split_dataset
from densesa.engine import compute_metrics, fit, get_device, predict, set_seed
from densesa.models import VARIANTS, build_model, count_parameters

FACTORS = ["Dilated", "ChannelAttn", "SpatialAttn"]


def effects(df, dataset):
    """Main effects (mean with factor on minus mean with it off) and
    two-way interactions, for accuracy, F1 and AUC."""
    rows = []
    for metric in ("Accuracy", "F1-Score", "AUC"):
        for f in FACTORS:
            effect = df.loc[df[f] == 1, metric].mean() - df.loc[df[f] == 0, metric].mean()
            rows.append({"Dataset": dataset, "Metric": metric, "Term": f,
                         "Type": "Main effect", "Effect": effect})
        for a, b in itertools.combinations(FACTORS, 2):
            cell = lambda i, j: df.loc[(df[a] == i) & (df[b] == j), metric].mean()
            rows.append({"Dataset": dataset, "Metric": metric, "Term": f"{a} x {b}",
                         "Type": "Interaction",
                         "Effect": 0.5 * (cell(1, 1) - cell(1, 0) - cell(0, 1) + cell(0, 0))})
    return pd.DataFrame(rows)


def main():
    args = add_common_args(argparse.ArgumentParser(description=__doc__)).parse_args()
    cfg = config_from_args(args)
    device = get_device()
    all_runs, all_effects = [], []

    for name in args.datasets:
        classes = DATASETS[name]["classes"]
        train, val, test, _ = split_dataset(load_dataset(args.data_root, name), cfg.seed)
        train_loader = make_loader(train, classes, cfg, augment=True)
        val_loader = make_loader(val, classes, cfg)
        test_loader = make_loader(test, classes, cfg)
        folder = Path(args.out) / name / "ablation"
        folder.mkdir(parents=True, exist_ok=True)
        runs = []

        for tag, label, d, c, s in VARIANTS:
            set_seed(cfg.seed)
            print(f"{name} {tag} {label}")
            model = build_model(tag, len(classes), cfg.pretrained).to(device)
            _, seconds = fit(model, train_loader, val_loader, cfg, device, verbose=False)
            labels, preds, probs, ms = predict(model, test_loader, device, cfg, warmup_batches=5)
            m = compute_metrics(labels, preds, probs, len(classes))
            m.pop("cm")
            torch.save(model.state_dict(), folder / f"{tag}.pt")
            runs.append({"Dataset": name, "Variant": tag, "Label": label,
                         "Dilated": int(d), "ChannelAttn": int(c), "SpatialAttn": int(s),
                         **m, "Params_M": count_parameters(model)[0] / 1e6,
                         "ms_per_image": ms, "Train_min": seconds / 60})
            print(f"  acc {m['Accuracy']:.4f}  f1 {m['F1-Score']:.4f}  auc {m['AUC']:.4f}")
            del model
            torch.cuda.empty_cache()

        df = pd.DataFrame(runs)
        eff = effects(df, name)
        plotting.factorial_bars(df, f"2^3 factorial ablation, {name}",
                                folder / "factorial_ablation.png", args.dpi)
        plotting.effects_plot(eff, f"Main effects and interactions, {name}",
                              folder / "factorial_effects.png", args.dpi)
        all_runs.append(df)
        all_effects.append(eff)

    out = Path(args.out)
    pd.concat(all_runs).to_csv(out / "factorial_ablation.csv", index=False)
    pd.concat(all_effects).to_csv(out / "factorial_effects.csv", index=False)


if __name__ == "__main__":
    main()
