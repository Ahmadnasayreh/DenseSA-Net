"""Dataset loading, the fixed train/val/test split and data loaders."""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import DATASETS

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def find_class_root(folder, classes):
    """Return the directory that directly holds the class folders.

    Some downloads wrap the class folders in one extra directory
    (e.g. Data1/bloodcells_dataset/), so we look one level down as well.
    """
    folder = Path(folder)
    wanted = set(classes)
    if wanted & {d.name for d in folder.iterdir() if d.is_dir()}:
        return folder
    for sub in folder.iterdir():
        if sub.is_dir() and wanted & {d.name for d in sub.iterdir() if d.is_dir()}:
            return sub
    raise FileNotFoundError(f"no class folders found under {folder}")


def load_dataset(data_root, name):
    """List every image of one dataset as a dataframe with path and label."""
    info = DATASETS[name]
    root = find_class_root(os.path.join(data_root, info["folder"]), info["classes"])
    rows = []
    for cls in info["classes"]:
        cdir = root / cls
        if not cdir.is_dir():
            print(f"warning: missing class folder {cdir}")
            continue
        for f in sorted(os.listdir(cdir)):
            if f.lower().endswith(IMAGE_EXT):
                rows.append({"path": str(cdir / f), "label": cls})
    return pd.DataFrame(rows)


def split_dataset(df, seed=42):
    """Stratified split: 10% test is held out first, then 80/10 train/val.

    The test partition is never used for training, model selection or
    cross-validation. Returns train, val, test and the 90% development set.
    """
    dev, test = train_test_split(df, test_size=0.10, stratify=df["label"],
                                 random_state=seed)
    train, val = train_test_split(dev, test_size=0.111, stratify=dev["label"],
                                  random_state=seed)
    return train, val, test, dev


def build_transform(img_size, augment):
    steps = [transforms.Resize((img_size, img_size))]
    if augment:
        steps += [transforms.RandomHorizontalFlip(p=0.5),
                  transforms.RandomRotation(10)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


class CellDataset(Dataset):
    def __init__(self, df, classes, transform):
        self.paths = df["path"].tolist()
        index = {c: i for i, c in enumerate(classes)}
        self.labels = [index[c] for c in df["label"]]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def make_loader(df, classes, cfg, augment=False):
    ds = CellDataset(df, classes, build_transform(cfg.img_size, augment))
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=augment,
                      num_workers=cfg.num_workers, pin_memory=torch.cuda.is_available(),
                      drop_last=augment)


def denormalize(tensor):
    """CHW normalised tensor -> HWC float image in [0, 1]."""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
