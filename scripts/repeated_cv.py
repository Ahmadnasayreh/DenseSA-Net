"""Repeated stratified cross-validation (3 x 7 folds) on the 90% development
set, plus the statistics of Tables 13-15 and Figure 12.

Each repeat uses its own partition seed. In the first repeat every
DenseSA-Net fold model is also evaluated on the held-out test set (Table 15).
Results are appended to a CSV after every fold, so an interrupted run can be
restarted with the same command and continues where it stopped.

    python scripts/repeated_cv.py --data-root data --out outputs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from densesa import plotting, stats
from densesa.config import CV_SEEDS, DATASETS, MODELS, N_FOLDS, add_common_args, config_from_args
from densesa.data import load_dataset, make_loader, split_dataset
from densesa.engine import fit, get_device, predict, set_seed
from densesa.models import build_model


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CV_SEEDS))
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    set_seed(cfg.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "cv_runs.csv"
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    def done(dataset, rep, fold, model):
        if log.empty:
            return False
        hit = log[(log.Dataset == dataset) & (log.Repeat == rep)
                  & (log.Fold == fold) & (log.Model == model)]
        return not hit.empty

    for name in args.datasets:
        classes = DATASETS[name]["classes"]
        _, _, test, dev = split_dataset(load_dataset(args.data_root, name), cfg.seed)
        test_loader = make_loader(test, classes, cfg)

        for rep, seed in enumerate(args.seeds, 1):
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            for fold, (tr_idx, va_idx) in enumerate(skf.split(dev, dev["label"]), 1):
                tr, va = dev.iloc[tr_idx], dev.iloc[va_idx]
                tr_loader = make_loader(tr, classes, cfg, augment=True)
                va_loader = make_loader(va, classes, cfg)
                for model_name in MODELS:
                    if done(name, rep, fold, model_name):
                        continue
                    model = build_model(model_name, len(classes), cfg.pretrained).to(device)
                    fit(model, tr_loader, va_loader, cfg, device, verbose=False)
                    labels, preds, _, _ = predict(model, va_loader, device, cfg)
                    row = {"Dataset": name, "Repeat": rep, "Seed": seed, "Fold": fold,
                           "Model": model_name, "Val_acc": (labels == preds).mean(),
                           "Test_acc": float("nan")}
                    if rep == 1 and model_name == "DenseSANet":
                        tl, tp, _, _ = predict(model, test_loader, device, cfg)
                        row["Test_acc"] = (tl == tp).mean()
                    log = pd.concat([log, pd.DataFrame([row])], ignore_index=True)
                    log.to_csv(log_path, index=False)
                    print(f"{name} repeat {rep} fold {fold} {model_name:10s} "
                          f"val {row['Val_acc']:.4f}")
                    del model
                    torch.cuda.empty_cache()

    summarise(log, args)


def summarise(log, args):
    out = Path(args.out)
    tests, omnibus, holdout = [], [], []
    for name, d in log.groupby("Dataset"):
        d = d.sort_values(["Repeat", "Fold"])
        scores = {m: d[d.Model == m]["Val_acc"].tolist() for m in MODELS
                  if (d.Model == m).any()}
        if len({len(v) for v in scores.values()}) != 1:
            print(f"{name}: unequal number of folds per model, run not complete")
            continue
        fr = stats.friedman(scores)
        if fr:
            omnibus.append({"Dataset": name, **fr})
        pw = stats.pairwise(scores)
        pw.insert(0, "Dataset", name)
        tests.append(pw)
        h = d[(d.Model == "DenseSANet") & (d.Repeat == 1)]["Test_acc"].dropna()
        holdout.append({"Dataset": name, "Mean": h.mean(), "SD": h.std(ddof=0),
                        "Min": h.min(), "Max": h.max(),
                        "Per_fold": ", ".join(f"{v:.4f}" for v in h)})
        pd.DataFrame(scores).to_csv(out / f"cv_folds_{name}.csv", index_label="Observation")
        n = len(next(iter(scores.values())))
        plotting.cv_boxplot(scores, f"Repeated cross-validation ({n} observations), {name}",
                            out / f"cv_boxplot_{name}.png", args.dpi)
    if omnibus:
        pd.DataFrame(omnibus).to_csv(out / "cv_friedman.csv", index=False)
    if tests:
        pd.concat(tests).to_csv(out / "cv_wilcoxon.csv", index=False)
    if holdout:
        pd.DataFrame(holdout).to_csv(out / "cv_holdout_test.csv", index=False)


if __name__ == "__main__":
    main()
