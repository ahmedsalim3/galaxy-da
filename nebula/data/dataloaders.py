from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from torch.utils.data import DataLoader
from torchvision import transforms

from nebula.data.dataset import (GalaxyDatasetSource, GalaxyDatasetTarget,
                                 split_dataset)
from nebula.data.normalization import compute_dataset_stats


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
            train_transform=self.src_transform,
            val_transform=self.src_transform,
            seed=self.seed,
        )
        if getattr(self, "tgt_dataset", None) is not None:
            self.train_tgt, self.val_tgt = split_dataset(
                self.tgt_dataset,
                val_size=self.val_size,
                train_transform=self.tgt_transform,
                val_transform=self.tgt_transform,
                seed=self.seed,
            )

    def _create_dataloaders(self):
        self.source_train_loader = DataLoader(
            self.train_src,
            batch_size=self.batch_size,
            shuffle=True,
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
    )

    train_loader = module.source_train_loader
    val_loader = module.source_test_loader
