import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from nebula.commons.logger import Logger

os.environ["LOG_LEVEL"] = str(10)


logger = Logger()


CLASSES = ("elliptical", "irregular", "spiral")


def get_label_mappings():
    """Return label to index and index to label mappings."""
    label2idx = {c: i for i, c in enumerate(CLASSES)}
    idx2label = {i: c for c, i in label2idx.items()}
    return label2idx, idx2label


class GalaxyDatasetSource(Dataset):
    """
    Source domain galaxy dataset with augmentations.

    Args:
        csv_or_json_file: Path to the CSV or JSON file containing the dataset metadata.
        img_dir: Path to the directory containing the images.
        transform: Transform to apply to the images.
        include_rotations: Whether to include rotations in the dataset.

        NOTE: include_rotations=True will include 8 augmented images for each original image.
        The augmented images are created by rotating the original image by 0, 90, 180, and 270 degrees and flipping the image horizontally.
    """

    def __init__(
        self,
        csv_or_json_file,
        img_dir,
        transform=None,
        include_rotations=True,
    ):
        csv_or_json_path = Path(csv_or_json_file)
        self.df = (
            pd.read_json(csv_or_json_path)
            if csv_or_json_path.suffix.lower() == ".json"
            else pd.read_csv(csv_or_json_path)
        )
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.rotations = [0, 90, 180, 270]
        self.include_rotations = include_rotations

        # Map string labels to integers
        self.label2idx, self.idx2label = get_label_mappings()
        self.df["label"] = self.df["classification"].str.strip().str.lower().map(self.label2idx)
        bad = self.df[self.df["label"].isna()]
        if len(bad):
            uniq = sorted(bad["classification"].astype(str).str.lower().unique().tolist())
            raise ValueError(f"Unknown class labels found: {uniq}. Expected one of {list(self.label2idx)}")
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self) -> int:
        if self.include_rotations:
            return len(self.df) * 8  # 4 rotations * 2 flips
        else:
            return len(self.df)

    def __getitem__(self, idx):
        if self.include_rotations:
            base_idx = idx // 8
            aug_idx = idx % 8

            row = self.df.iloc[base_idx]
            img_path = self.img_dir / f"subhalo_{row['subhalo_id']}.png"
            label = int(row["label"])
            image = Image.open(img_path).convert("RGB")

            rotation = self.rotations[aug_idx % 4]
            flip = (aug_idx // 4) == 1

            image = image.rotate(rotation)
            if flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            row = self.df.iloc[idx]
            img_path = self.img_dir / f"subhalo_{row['subhalo_id']}.png"
            image = Image.open(img_path).convert("RGB")
            label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


class GalaxyDatasetTarget(Dataset):
    """Target domain galaxy dataset."""

    def __init__(self, csv_or_json_file: Path, img_dir: Path, transform=None):
        csv_or_json_path = Path(csv_or_json_file)
        self.df = (
            pd.read_json(csv_or_json_path)
            if csv_or_json_path.suffix.lower() == ".json"
            else pd.read_csv(csv_or_json_path)
        )
        self.img_dir = Path(img_dir)
        self.transform = transform

        # Map string labels to integers
        self.label2idx, self.idx2label = get_label_mappings()
        self.df["label"] = self.df["classification"].str.strip().str.lower().map(self.label2idx)
        bad = self.df[self.df["label"].isna()]
        if len(bad):
            uniq = sorted(bad["classification"].astype(str).str.lower().unique().tolist())
            raise ValueError(f"Unknown class labels found: {uniq}. Expected one of {list(self.label2idx)}")
        self.df["label"] = self.df["label"].astype(int)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["subhalo_id"]

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label

class TransformSubset(Dataset):
    def __init__(self, base_dataset: Dataset, indices, transform=None):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=int)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.base[self.indices[i]]
        if self.transform:
            img = self.transform(img)
        return img, label


def split_dataset(
    dataset, val_size=0.2, train_transform=None, val_transform=None, seed=42
):
    """
    Stratified train/val split maintaining class proportions.
    The idea is to split the dataset into train and val sets while maintaining the class proportions.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Get base labels (handle rotation augmentation for source)
    has_rot = getattr(dataset, "include_rotations", False)
    base_labels = dataset.df["label"].values

    # Group indices by class
    train_idx, val_idx = [], []
    class_stats = []
    AUG = 8 if has_rot else 1

    for cls in np.unique(base_labels):
        cls_indices = np.where(base_labels == cls)[0]
        np.random.shuffle(cls_indices)

        # Split: val_size per class
        n_val = max(1, int(len(cls_indices) * val_size))
        cls_val_base = cls_indices[:n_val]
        cls_train_base = cls_indices[n_val:]

        # Expand if rotations (1 base → 8 augmented)
        if has_rot:
            def expand(idxs):
                if len(idxs) == 0:
                    return np.empty(0, dtype=int)
                blocks = [np.arange(i * AUG, (i + 1) * AUG, dtype=int) for i in idxs]
                return np.concatenate(blocks) if blocks else np.empty(0, dtype=int)

            cls_val = expand(cls_val_base)
            cls_train = expand(cls_train_base)

        else:
            cls_val = cls_val_base
            cls_train = cls_train_base

        train_idx.extend(cls_train.tolist())
        val_idx.extend(cls_val.tolist())

        # Store stats for clean table
        class_stats.append({
            "class": CLASSES[cls],
            "train_aug": len(cls_train),
            "val_aug": len(cls_val),
            "total_base": len(cls_indices),
        })

    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)

    train_subset = TransformSubset(dataset, train_idx, transform=train_transform)
    val_subset = TransformSubset(dataset, val_idx, transform=val_transform)

    # Print clean table
    logger.debug("┌─────────────┬──────────┬──────────┬─────────────┐")
    logger.debug("│ Class       │ Train*   │ Val*     │ Total(base) │")
    logger.debug("├─────────────┼──────────┼──────────┼─────────────┤")

    for stats in class_stats:
        logger.debug(
            f"│ {stats['class']:<11} │ {stats['train_aug']:>8} │ {stats['val_aug']:>8} │ {stats['total_base']:>11} │"
        )

    logger.debug("├─────────────┼──────────┼──────────┼─────────────┤")
    logger.debug(
        f"│ Total       │ {len(train_subset):>8} │ {len(val_subset):>8} │ {len(base_labels):>11} │"
    )
    logger.debug("└─────────────┴──────────┴──────────┴─────────────┘")
    logger.debug("* augmented counts when include_rotations=True")

    return train_subset, val_subset


if __name__ == "__main__":
    from torchvision import transforms

    # Source dataset AUGMENTED
    src_dataset = GalaxyDatasetSource(
        Path("data/source/source_galaxy_labels.csv"),
        Path("data/source/galaxy_images_rgb"),
        transform=transforms.ToTensor(),
        include_rotations=True,
    )
    train_src, val_src = split_dataset(src_dataset, val_size=0.2, seed=42)

    # Target dataset DESI
    logger.debug("Loading target dataset DESI...")
    tgt_dataset = GalaxyDatasetTarget(
        Path("data/target/gz_desi_labels.csv"),
        Path("data/target/gz_desi"),
        transform=transforms.ToTensor(),
    )
    train_tgt, val_tgt = split_dataset(tgt_dataset, val_size=0.2, seed=42)

    # Target dataset GZ2
    logger.debug("Loading target dataset GZ2...")
    gz2_tgt_dataset = GalaxyDatasetTarget(
        Path("data/target/gz2_galaxy_labels.csv"),
        Path("data/target/gz2_images"),
        transform=transforms.ToTensor(),
    )
    train_gz2_tgt, val_gz2_tgt = split_dataset(gz2_tgt_dataset, val_size=0.2, seed=42)
