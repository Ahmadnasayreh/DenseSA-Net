"""Grad-CAM and spatial attention maps, failure cases, and the quantitative
check of attention against nuclear regions (Table 16, Figures 13-15).

Needs the DenseSA-Net weights produced by train_final.py.

    python scripts/explainability.py --data-root data --out outputs
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

from densesa import plotting
from densesa.config import DATASETS, add_common_args, config_from_args
from densesa.data import denormalize, load_dataset, make_loader, split_dataset
from densesa.engine import get_device
from densesa.explain import GradCAM, alignment_records, summarise_alignment, upsample
from densesa.models import DenseSANet


def maps(model, cam, image, class_idx):
    rgb = denormalize(image)
    heat, _ = cam(image, class_idx)
    with torch.no_grad():
        model(image.unsqueeze(0).to(next(model.parameters()).device))
    att = model.last_attention[0, 0].cpu().numpy()
    size = rgb.shape[:2]
    return rgb, upsample(heat, size), upsample(att, size)


def main():
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--max-failures", type=int, default=6)
    args = parser.parse_args()
    cfg = config_from_args(args)
    device = get_device()
    summaries = []

    for name in args.datasets:
        classes = DATASETS[name]["classes"]
        _, _, test, _ = split_dataset(load_dataset(args.data_root, name), cfg.seed)
        loader = make_loader(test, classes, cfg)
        folder = Path(args.out) / name
        model = DenseSANet(len(classes), pretrained=False)
        model.load_state_dict(torch.load(folder / "DenseSANet.pt", map_location="cpu"))
        model.to(device).eval()
        cam = GradCAM(model)

        # one correctly classified example per class, plus every error
        examples, errors = {}, []
        for x, y in loader:
            with torch.no_grad():
                prob = torch.softmax(model(x.to(device)).float(), 1).cpu()
            for i in range(x.size(0)):
                t, p = int(y[i]), int(prob[i].argmax())
                if t == p and t not in examples:
                    examples[t] = x[i]
                elif t != p:
                    errors.append((float(prob[i, p]), x[i], t, p))

        rows = []
        for c in sorted(examples):
            rgb, heat, att = maps(model, cam, examples[c], c)
            rows.append((rgb, heat, att, classes[c]))
        plotting.saliency_rows(rows, f"Grad-CAM and spatial attention, {name}",
                               folder / "xai_maps.png", args.dpi)

        if errors:
            errors.sort(key=lambda e: -e[0])
            rows = []
            for conf, img, t, p in errors[:args.max_failures]:
                rgb, heat, att = maps(model, cam, img, p)
                rows.append((rgb, heat, att,
                             f"True: {classes[t]}\nPredicted: {classes[p]} ({conf:.3f})"))
            plotting.saliency_rows(rows, f"Most confident errors, {name} "
                                         f"({len(errors)} of {len(test)} misclassified)",
                                   folder / "failure_cases.png", args.dpi,
                                   title_color="#B91C1C")
        else:
            print(f"{name}: no misclassified test images")
        cam.remove()

        records = alignment_records(model, loader, classes, name, args.max_samples)
        records.to_csv(folder / "alignment_per_image.csv", index=False)
        s = summarise_alignment(records, name)
        summaries.append(s)
        print(f"{name}: nucleus covers {s['mask_fraction'] * 100:.1f}%  "
              f"pointing lift {s['PointingGame_lift']:.2f}x  "
              f"enrichment {s['Enrichment']:.2f}x  IoU lift {s['IoU_lift']:.2f}x")
        del model
        torch.cuda.empty_cache()

    summary = pd.DataFrame(summaries)
    summary.to_csv(Path(args.out) / "alignment_summary.csv", index=False)
    plotting.alignment_vs_chance(summary, Path(args.out) / "alignment_vs_chance.png", args.dpi)


if __name__ == "__main__":
    main()
