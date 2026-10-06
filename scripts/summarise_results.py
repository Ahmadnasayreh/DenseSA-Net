"""Recompute the statistical tables from the result files in results/.

No training or GPU needed. Reproduces Tables 13 and 14 (repeated
cross-validation, Friedman and Wilcoxon tests) and Table 16 (alignment of
Grad-CAM with nuclear regions against chance).

    python scripts/summarise_results.py
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from densesa import stats
from densesa.explain import summarise_alignment

pd.set_option("display.width", 200)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(Path(__file__).resolve().parents[1] / "results"))
    args = p.parse_args()
    root = Path(args.results)
    datasets = ("Dataset1", "Dataset2", "Dataset3")

    print("Repeated cross-validation (mean +/- SD of fold accuracy)")
    for ds in datasets:
        f = pd.read_csv(root / f"cv_folds_{ds}.csv")
        line = "  ".join(f"{m} {f[m].mean():.4f} +/- {f[m].std():.4f}"
                         for m in ("DenseSANet", "VGG16", "CustomCNN"))
        print(f"  {ds}: {line}")

    print("\nFriedman test")
    for ds in datasets:
        f = pd.read_csv(root / f"cv_folds_{ds}.csv")
        r = stats.friedman({m: f[m] for m in ("DenseSANet", "VGG16", "CustomCNN")})
        print(f"  {ds}: chi2 = {r['Friedman_chi2']:.2f}  p = {r['p_value']:.1e}"
              f"  (n = {r['n_observations']}, max chi2 = {r['theoretical_max_chi2']})")

    print("\nWilcoxon signed-rank tests, effect sizes, bootstrap 95% CI")
    rows = []
    for ds in datasets:
        f = pd.read_csv(root / f"cv_folds_{ds}.csv")
        t = stats.pairwise({m: f[m] for m in ("DenseSANet", "VGG16", "CustomCNN")})
        t.insert(0, "Dataset", ds)
        rows.append(t)
    t = pd.concat(rows)
    cols = ["Dataset", "Comparison", "Subset", "n", "Wilcoxon_W", "p_value",
            "Cliffs_delta", "Mean_diff", "CI95_low", "CI95_high", "Folds_won"]
    print(t[cols].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\nGrad-CAM alignment with nuclear regions (observed / chance)")
    for ds in datasets:
        s = summarise_alignment(pd.read_csv(root / f"alignment_per_image_{ds}.csv"), ds)
        print(f"  {ds}: nucleus area {s['mask_fraction'] * 100:.1f}%  "
              f"pointing {s['PointingGame']:.3f}/{s['PointingGame_chance']:.3f} "
              f"({s['PointingGame_lift']:.2f}x)  "
              f"energy {s['EnergyInMask']:.3f}/{s['EnergyInMask_chance']:.3f} "
              f"({s['Enrichment']:.2f}x)  "
              f"IoU {s['IoU_topk']:.3f}/{s['IoU_chance']:.3f} ({s['IoU_lift']:.2f}x)")


if __name__ == "__main__":
    main()
