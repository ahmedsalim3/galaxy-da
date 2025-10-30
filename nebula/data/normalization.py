# Per-channel z-score normalization
# =================================
#
# This script computes the per-channel mean and standard deviation of an image dataset.
# It is used to normalize the images to have a mean of 0 and a standard deviation of 1.
#
# The formula for z-score normalization is:
#
# Z-score normalization means that for each RGB channel (c):
# z = (x - μ_c) / σ_c
#
# where:
# - x is the pixel value,
# - μ_c is the mean pixel value of channel (c) across the entire dataset,
# - σ_c is the standard deviation of channel (c) across the entire dataset.


import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from nebula.commons.logger import Logger

os.environ["LOG_LEVEL"] = str(10)


logger = Logger()


def compute_dataset_stats(dataset, batch_size=32, num_workers=4):
    """Compute mean and std for normalization."""
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    mean = torch.zeros(3)
    sumsq = torch.zeros(3)
    num_pixels = 0

    for images, _ in tqdm(
        loader, desc=f"Computing mean/std for {dataset.__class__.__name__}", leave=False
    ):
        B, C, H, W = images.shape
        num_pixels += B * H * W
        mean += images.sum(dim=[0, 2, 3])
        sumsq += (images**2).sum(dim=[0, 2, 3])

    mean /= num_pixels
    std = torch.sqrt(sumsq / num_pixels - mean**2)

    return mean, std


if __name__ == "__main__":
    from pathlib import Path

    from torchvision import transforms

    from nebula.data.dataset import GalaxyDatasetSource, GalaxyDatasetTarget

    dataset = GalaxyDatasetSource(
        csv_or_json_file=Path("data/source/source_galaxy_labels.csv"),
        img_dir=Path("data/source/galaxy_images_rgb"),
        transform=transforms.ToTensor(),
        include_rotations=False,
    )
    mean, std = compute_dataset_stats(dataset)
    logger.debug(f"SRC Mean: {mean.tolist()}")
    logger.debug(f"SRC Std: {std.tolist()}")

    dataset = GalaxyDatasetTarget(
        csv_or_json_file=Path("data/target/gz2_galaxy_labels.csv"),
        img_dir=Path("data/target/gz2_images"),
        transform=transforms.ToTensor(),
    )
    mean, std = compute_dataset_stats(dataset)
    logger.debug(f"TGT Mean: {mean.tolist()}")
    logger.debug(f"TGT Std: {std.tolist()}")
