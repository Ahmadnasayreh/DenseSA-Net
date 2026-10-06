"""Grad-CAM and the quantitative check of attention against nuclear regions.

A nucleus reference mask is obtained from the blue-ratio image with an Otsu
threshold. Each measure is reported together with the value a random
saliency map would get, so that 1.0 means "no better than chance".
"""
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import mannwhitneyu

from .data import denormalize


class GradCAM:
    """Grad-CAM on the last dense block of a DenseNet-121 backbone."""

    def __init__(self, model, layer=None):
        self.model = model
        layer = layer or model.features.denseblock4
        self.acts = self.grads = None
        self._h = [layer.register_forward_hook(self._save_act),
                   layer.register_full_backward_hook(self._save_grad)]

    def _save_act(self, module, inp, out):
        self.acts = out.detach()

    def _save_grad(self, module, gin, gout):
        self.grads = gout[0].detach()

    def __call__(self, image, class_idx=None):
        self.model.eval()
        device = next(self.model.parameters()).device
        x = image.unsqueeze(0).to(device).requires_grad_(True)
        out = self.model(x)
        if class_idx is None:
            class_idx = int(out.argmax(1))
        self.model.zero_grad()
        out[0, class_idx].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1)).squeeze(0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx

    def remove(self):
        for h in self._h:
            h.remove()


def upsample(heat, size):
    """Bilinear resize of a small map to (H, W), rescaled to [0, 1]."""
    h = F.interpolate(torch.as_tensor(heat)[None, None].float(), size=size,
                      mode="bilinear", align_corners=False)[0, 0].numpy()
    return (h - h.min()) / (h.max() - h.min() + 1e-8)


def nucleus_mask(rgb):
    """Blue-ratio transform followed by an Otsu threshold.

    BR = 100 B / (1 + R + G) * 256 / (1 + B + R + G)
    """
    img = rgb.astype(np.float32) * 255
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    br = (100 * b) / (1 + r + g) * (256 / (1 + b + r + g))
    br = (br - br.min()) / (br.max() - br.min() + 1e-8)
    hist, edges = np.histogram(br.ravel(), bins=256, range=(0, 1))
    hist = hist.astype(float) / hist.sum()
    mids = (edges[:-1] + edges[1:]) / 2
    w = np.cumsum(hist)
    mu = np.cumsum(hist * mids)
    denom = w * (1 - w)
    denom[denom == 0] = 1e-12
    between = (mu[-1] * w - mu) ** 2 / denom
    return br > mids[int(between.argmax())]


def cam_entropy(cam):
    p = cam.ravel().astype(np.float64)
    p = p - p.min()
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(cam.size))


def alignment_records(model, loader, classes, dataset, max_samples=400, topk=0.20):
    """Per-image alignment measures between Grad-CAM and the nucleus mask."""
    cam = GradCAM(model)
    device = next(model.parameters()).device
    records, seen = [], 0
    for x, y in loader:
        with torch.no_grad():
            preds = model(x.to(device)).argmax(1).cpu()
        for i in range(x.size(0)):
            if seen >= max_samples:
                break
            seen += 1
            rgb = denormalize(x[i])
            heat, pred = cam(x[i], int(preds[i]))
            heat = upsample(heat, rgb.shape[:2])
            mask = nucleus_mask(rgb)
            if mask.sum() < 20:
                continue
            py, px = np.unravel_index(heat.argmax(), heat.shape)
            top = heat >= np.quantile(heat, 1 - topk)
            union = np.logical_or(top, mask).sum()
            records.append({
                "Dataset": dataset, "true": classes[int(y[i])], "pred": classes[pred],
                "correct": int(int(y[i]) == pred),
                "pointing_hit": int(mask[py, px]),
                "energy_in_mask": float(heat[mask].sum() / (heat.sum() + 1e-8)),
                "iou_topk": float(np.logical_and(top, mask).sum() / union) if union else 0.0,
                "cam_entropy": cam_entropy(heat),
                "mask_fraction": float(mask.mean()),
            })
        if seen >= max_samples:
            break
    cam.remove()
    return pd.DataFrame(records)


def summarise_alignment(df, dataset, topk=0.20, min_errors=5):
    """Each measure next to its chance value.

    For a nucleus covering a fraction m of the image, a uniform random map
    scores m on the pointing game and on energy-in-mask, and k*m/(k+m-k*m)
    on the IoU of its top-k region.
    """
    m = df["mask_fraction"].mean()
    pg, en, io = df["pointing_hit"].mean(), df["energy_in_mask"].mean(), df["iou_topk"].mean()
    iou_chance = topk * m / (topk + m - topk * m)
    out = {"Dataset": dataset, "n_samples": len(df),
           "n_correct": int(df["correct"].sum()),
           "n_incorrect": int((df["correct"] == 0).sum()),
           "mask_fraction": m,
           "PointingGame": pg, "PointingGame_chance": m, "PointingGame_lift": pg / m,
           "EnergyInMask": en, "EnergyInMask_chance": m, "Enrichment": en / m,
           "IoU_topk": io, "IoU_chance": iou_chance, "IoU_lift": io / iou_chance,
           "Entropy": df["cam_entropy"].mean()}
    wrong = df[df["correct"] == 0]
    right = df[df["correct"] == 1]
    if len(wrong) >= min_errors:
        for col in ("energy_in_mask", "iou_topk", "cam_entropy"):
            out[f"MWU_p_{col}"] = float(mannwhitneyu(right[col], wrong[col]).pvalue)
    else:
        out["MWU_note"] = (f"not computed: {len(wrong)} misclassified sample(s), "
                           "too few for a meaningful test")
    return out
