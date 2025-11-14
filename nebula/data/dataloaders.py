from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from nebula.commons.logger import Logger
from nebula.data.dataset import (CLASSES, GalaxyDatasetSource,
                                 GalaxyDatasetTarget, split_dataset)
from nebula.data.normalization import compute_dataset_stats

logger = Logger()


@dataclass
class GalaxyDataModule:
    source_img_dir: Path
    source_labels: Path
    target_img_dir: Path
    target_labels: Path
    include_rotations: bool = False  # for source dataset

    shared_norm: bool = False  # if True, normalize both datasets using source stats
    source_mean: Optional[List[float]] = None
    source_std: Optional[List[float]] = None
    target_mean: Optional[List[float]] = None
    target_std: Optional[List[float]] = None

    image_size: Tuple[int, int] = (28, 28)
    batch_size: int = 64
    val_size: float = 0.2
    num_workers: int = 4
    seed: int = 42
    use_sampler: bool = False
    sampler_smoothing: float = 0.0
    sampler_power: float = 1.0
    sampler_replacement: bool = True

    src_dataset: GalaxyDatasetSource = field(init=False)
    tgt_dataset: Optional[GalaxyDatasetTarget] = field(init=False, default=None)
    train_tgt: Optional[GalaxyDatasetTarget] = field(init=False, default=None)
    val_tgt: Optional[GalaxyDatasetTarget] = field(init=False, default=None)
    source_train_loader: DataLoader = field(init=False)
    source_test_loader: DataLoader = field(init=False)
    target_train_loader: Optional[DataLoader] = field(init=False, default=None)
    target_test_loader: Optional[DataLoader] = field(init=False, default=None)

    def __post_init__(self):
        self._compute_stats()
        self._setup_transforms()
        self._split_datasets()
        self._create_dataloaders()

    def _compute_stats(self):
        if self.source_mean is None or self.source_std is None:
            self.raw_src = GalaxyDatasetSource(
                self.source_labels,
                self.source_img_dir,
                transform=transforms.Compose(
                    [
                        transforms.Resize(
                            self.image_size,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                        transforms.ToTensor(),
                    ]
                ),
                include_rotations=self.include_rotations,
            )
            self.source_mean, self.source_std = compute_dataset_stats(
                self.raw_src, batch_size=self.batch_size, num_workers=self.num_workers
            )
        if (
            not self.shared_norm
            and (self.target_mean is None or self.target_std is None)
            and (self.target_img_dir is not None or self.target_labels is not None)
        ):
            self.raw_tgt = GalaxyDatasetTarget(
                self.target_labels,
                self.target_img_dir,
                transform=transforms.Compose(
                    [
                        transforms.Resize(
                            self.image_size,
                            interpolation=transforms.InterpolationMode.BILINEAR,
                            antialias=True,
                        ),
                        transforms.ToTensor(),
                    ]
                ),
            )
            self.target_mean, self.target_std = compute_dataset_stats(
                self.raw_tgt, batch_size=self.batch_size, num_workers=self.num_workers
            )
        if self.shared_norm:
            self.target_mean, self.target_std = self.source_mean, self.source_std

    def _setup_transforms(self):
        self.src_transform = transforms.Compose(
            [
                transforms.Resize(
                    self.image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.source_mean, std=self.source_std),
            ]
        )
        if self.target_mean is not None and self.target_std is not None:
            self.tgt_transform = transforms.Compose(
                [
                    transforms.Resize(
                        self.image_size,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.target_mean, std=self.target_std),
                ]
            )
        # Create normalized datasets
        self.src_dataset = GalaxyDatasetSource(
            self.source_labels,
            self.source_img_dir,
            transform=self.src_transform,
            include_rotations=self.include_rotations,
        )
        if (
            self.target_img_dir is not None or self.target_labels is not None
        ) and hasattr(self, "tgt_transform"):
            self.tgt_dataset = GalaxyDatasetTarget(
                self.target_labels, self.target_img_dir, transform=self.tgt_transform
            )

    def _split_datasets(self):
        self.train_src, self.val_src = split_dataset(
            self.src_dataset,
            val_size=self.val_size,
            seed=self.seed,
        )
        if getattr(self, "tgt_dataset", None) is not None:
            self.train_tgt, self.val_tgt = split_dataset(
                self.tgt_dataset,
                val_size=self.val_size,
                seed=self.seed,
            )

    def _create_dataloaders(self):
        if self.use_sampler:
            logger.debug("Using WeightedRandomSampler for source training data:")

            all_labels = self.train_src.dataset.labels_tensor
            train_indices = self.train_src.indices
            labels = all_labels[train_indices]
            # ========================

            num_classes = int(labels.max().item()) + 1
            class_sample_count = torch.tensor(
                [(labels == i).sum().item() for i in range(num_classes)]
            )

            # Compute weights based on smoothing and power settings
            # - smoothing=0, power=1.0: old behavior (1.0 / count)
            # - smoothing=0, power!=1.0: use power without smoothing (1.0 / count ** power)
            # - smoothing>0: use both smoothing and power (1.0 / (count + smoothing) ** power)
            if self.sampler_smoothing > 0:
                # With smoothing: prevents extreme weights for very small classes
                smoothed_counts = class_sample_count.float() + self.sampler_smoothing
                weights = 1.0 / (smoothed_counts**self.sampler_power)
                logger.debug(
                    f" - Sampler config: smoothing={self.sampler_smoothing}, power={self.sampler_power}"
                )
            elif self.sampler_power != 1.0:
                weights = 1.0 / (class_sample_count.float() ** self.sampler_power)
                logger.debug(f" - Sampler config: power={self.sampler_power}")
            else:
                # simple inverse frequency (smoothing=0, power=1.0)
                weights = 1.0 / class_sample_count.float()

            sample_weights = weights[labels]

            # Log class distribution and weights
            logger.debug(f" - Total training samples: {len(labels)}")

            for i in range(num_classes):
                class_name = CLASSES[i] if i < len(CLASSES) else f"class_{i}"
                count = class_sample_count[i].item()
                weight = weights[i].item()
                logger.debug(f"  {class_name}: {count} samples, weight: {weight:.6f}")

            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=self.sampler_replacement,
            )
            logger.debug(
                f"Created WeightedRandomSampler with {len(sample_weights)} samples, replacement={self.sampler_replacement}"
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
            logger.debug("Using standard DataLoader with shuffle=True (no sampler)")

        self.source_train_loader = DataLoader(
            self.train_src,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
        )
        self.source_test_loader = DataLoader(
            self.val_src,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        if getattr(self, "train_tgt", None) is not None:
            self.target_train_loader = DataLoader(
                self.train_tgt,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
        if getattr(self, "val_tgt", None) is not None:
            self.target_test_loader = DataLoader(
                self.val_tgt,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )


if __name__ == "__main__":
    module = GalaxyDataModule(
        source_img_dir=Path("data/source/galaxy_images_rgb"),
        source_labels=Path("data/source/source_galaxy_labels.csv"),
        target_img_dir=Path("data/target/gz2_images"),
        target_labels=Path("data/target/gz2_galaxy_labels.csv"),
        use_sampler=True,
    )

    train_loader = module.source_train_loader
    val_loader = module.source_test_loader
