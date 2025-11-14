from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as escnn_nn


@dataclass
class ESCNNConfig:
    """Configuration for E(2)-equivariant steerable CNNs.

    This setup follows the design principles from Weiler & Cesa's NeurIPS 2019 paper
    "General E(2)-Equivariant Steerable CNNs" (https://arxiv.org/pdf/1911.08251).

    Attributes:
        num_classes: Target classes for classification
        group: Which symmetry group to use (e.g., "C8" for 8-way rotations, "D4" for
            4-way rotations + reflections). Check Table 1 in the paper for all options.
        N: How many discrete rotations in the group (8 means 45° increments)
        base_width: Number of regular-representation feature fields in the first layer.
            Deeper layers automatically get wider (2x, then 4x this value).
        dropout: Dropout probability in the final classifier
    """

    num_classes: int = 3
    group: str = "C8"  # Symmetry group label (e.g., "C8", "D4")
    N: int = 8  # Order of the group (number of rotations)
    base_width: int = 16  # Number of feature fields in the first equivariant layer
    dropout: float = 0.3  # Dropout rate in the classifier head


def _build_group(group: str = "C8", N: int = 8):
    """Helper to construct the right symmetry group from a string label.

    We support cyclic groups (C_N - just rotations) and dihedral groups (D_N -
    rotations plus reflections). The paper discusses these in Section 2.1.
    """
    g = group.upper()
    if g.startswith("D"):
        return gspaces.flipRot2dOnR2(N=N)
    return gspaces.rot2dOnR2(N=N)


class EqvConvBlock(nn.Module):
    """A single equivariant convolutional block: conv → batch norm → ReLU.

    This is the workhorse building block of our steerable CNNs. Unlike vanilla convs,
    these operations respect the group symmetry: if you rotate the input, the features
    rotate accordingly. The math behind this comes from the steerability constraint
    described in Section 2.4-2.5 of Weiler & Cesa (https://arxiv.org/pdf/1911.08251).

    The key insight: by constraining kernel shapes through group theory, we guarantee
    that feature transformations compose properly. When the input undergoes g ∈ G,
    the output transforms via the output representation ρ_out(g). This constraint
    (Equation 2 in the paper) is what R2Conv implements under the hood.

    The actual mechanics work like this:
        - R2Conv uses basis kernels that satisfy k(gx) = ρ_out(g) k(x) ρ_in(g)^{-1}
        - InnerBatchNorm normalizes within each "fiber" (each copy of the representation)
        - ReLU is applied in a way that preserves equivariance

    Args:
        in_type: Describes how input features transform (their "field type")
        out_type: Describes how output features should transform
        kernel_size: Spatial extent of the convolutional kernel
        stride: Downsampling factor (stride > 1 reduces spatial resolution)
        padding: Border padding; defaults to kernel_size//2 for "same" padding

    Input/Output:
        Takes a GeometricTensor (features + their transformation law) and returns
        another GeometricTensor with the new transformation law specified by out_type.
    """

    def __init__(
        self,
        in_type: "escnn_nn.FieldType",
        out_type: "escnn_nn.FieldType",
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | None = None,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = escnn_nn.R2Conv(
            in_type,
            out_type,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = escnn_nn.InnerBatchNorm(out_type)
        self.act = escnn_nn.ReLU(out_type, inplace=True)

    def forward(self, x: "escnn_nn.GeometricTensor") -> "escnn_nn.GeometricTensor":
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ESCNNSteerable(nn.Module):
    """E(2)-steerable CNN encoder using group-equivariant convolutions.

    This model is built on the foundation laid out in Weiler & Cesa's 2019 work
    (https://arxiv.org/pdf/1911.08251). The core idea is beautifully simple: natural
    images contain patterns at all orientations, so why not build that symmetry directly
    into the network architecture?

    What makes this work:
    ----------------------
    Traditional CNNs only understand translations - slide an object left/right/up/down
    and the features shift accordingly. But rotate that object 45°? The network has to
    relearn the same pattern. That's wasteful.

    This architecture understands rotations (and optionally reflections) as first-class
    citizens. When you rotate the input image, the internal features rotate in a
    mathematically precise way. The paper calls this "steerable" because features carry
    orientation information and steer appropriately under transformations.

    How it's structured:
    --------------------
    We stack three equivariant conv blocks with progressive downsampling (via stride=2).
    Each block widens the features: base_width → 2×base_width → 4×base_width channels.
    After spatial pooling, we use GroupPooling to extract rotation/reflection-invariant
    descriptors. These go into a standard MLP classifier.

    The architecture diagram from Section 2.6:
        RGB Input (3 scalar fields)
                ↓
        EqvConvBlock (base_width regular fields)
                ↓ (stride 2)
        EqvConvBlock (2× base_width)
                ↓ (stride 2)
        EqvConvBlock (4× base_width)
                ↓
        Spatial Global Average Pooling
                ↓
        Group Pooling (→ invariant features)
                ↓
        MLP Head → class logits

    Theory highlights (for the curious):
    ------------------------------------
    Section 2.1 discusses how we pick symmetry groups: C_N (N rotations), D_N (N
    rotations + N reflections), or even continuous SO(2)/O(2). We implement the
    discrete versions here for practical efficiency.

    Section 2.2 introduces "feature fields" - think of them as vector/tensor fields
    over the image plane. Each pixel doesn't just have a scalar value; it has a
    small vector that transforms under rotations. The "FieldType" specifies how.

    Section 2.5 solves the kernel constraint analytically using harmonic decomposition.
    This is where the magic happens - we get provably correct equivariant convolutions
    with fewer parameters than unconstrained kernels would need.

    Practical notes:
    ----------------
    The model expects standard PyTorch tensors [B, 3, H, W] as input and returns
    (logits, features) to match typical classification interfaces. The features are
    rotation-invariant by construction, making them great for downstream tasks.

    Args:
        config: ESCNNConfig dataclass with all the knobs (group type, width, etc.)

    Returns:
        logits: [B, num_classes] - raw classification scores
        features: [B, feature_dim] - invariant feature vector after group pooling
    """

    def __init__(
        self,
        config: ESCNNConfig,
    ) -> None:
        super().__init__()

        # Build the symmetry group (C_N or D_N)
        self.r2_act = _build_group(config.group, config.N)

        # Set up representation types we'll use
        tr = self.r2_act.trivial_repr  # Scalar fields (like grayscale pixels)
        rr = self.r2_act.regular_repr  # Regular representation (full group orbit)

        # Input: RGB image = 3 scalar fields
        self.in_type = escnn_nn.FieldType(self.r2_act, [tr] * 3)

        # Define field types for each layer (progressively wider)
        ft1 = escnn_nn.FieldType(self.r2_act, [rr] * config.base_width)
        ft2 = escnn_nn.FieldType(self.r2_act, [rr] * (config.base_width * 2))
        ft3 = escnn_nn.FieldType(self.r2_act, [rr] * (config.base_width * 4))

        # Deeper architecture: 4 conv blocks for better feature extraction
        # Original: 3 blocks, now: 4 blocks with additional capacity
        self.block1 = EqvConvBlock(self.in_type, ft1, kernel_size=5, stride=2)
        self.block2 = EqvConvBlock(ft1, ft2, kernel_size=3, stride=2)
        self.block3 = EqvConvBlock(
            ft2, ft3, kernel_size=3, stride=1
        )  # No stride - keep spatial info
        # Add 4th block with same width for more depth
        self.block4 = EqvConvBlock(
            ft3, ft3, kernel_size=3, stride=1
        )  # Additional depth

        # Group pooling: projects to trivial representation (rotation invariance)
        self.gpool = escnn_nn.GroupPooling(ft3)

        # After group pooling, we get scalars - one per field in ft3
        self.feature_dim = self.gpool.out_type.size

        # Larger MLP head for classification - better capacity for domain adaptation
        # Original: Linear(feature_dim → 128 → 3)
        # New: Linear(feature_dim → 256 → 128 → 3) with dropout
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout * 0.5),  # Less dropout in second layer
            nn.Linear(128, config.num_classes),
        )

        self.class_scales = nn.Parameter(torch.ones(config.num_classes))

    def forward(self, x: torch.Tensor):
        """Forward pass through the equivariant encoder.

        Args:
            x: Input images [B, 3, H, W]

        Returns:
            logits: Class preds [B, num_classes]
            feats: Invariant feature vector [B, feature_dim] - useful for domain
                adaptation, transfer learning, or embedding-based tasks

        Implementation notes:
        ---------------------
        We first wrap the input tensor as a GeometricTensor so the lib knows it's
        made of three scalar fields. Then we pass through equivariant blocks, maintaining
        the proper transformation behavior at each step

        Spatial pooling (adaptive avg pool to 1x1) removes positional dependence while
        staying equivariant. Then GroupPooling aggregates over group elements - for a
        C8 regular field, this means averaging over all 8 rotation angles. The result
        is provably invariant to rotations/reflections.

        Why this works: The paper shows (Section 2.2) that mapping to the trivial
        representation (what GroupPooling does) is the canonical way to get invariants.
        Think of it as "forgetting" the orientation information while keeping what's
        orientation-independent.
        """
        # Wrap input as geometric tensor with trivial field type
        x = escnn_nn.GeometricTensor(x, self.in_type)

        # Equivariant feature extraction (deeper architecture)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # Additional depth

        # Spatial pooling: remove spatial dims while staying equivariant
        t = x.tensor
        t = F.adaptive_avg_pool2d(t, output_size=1)
        x = escnn_nn.GeometricTensor(t, x.type)

        # Group pooling: from equivariant features to invariant features
        x = self.gpool(x)

        # Flatten and classify
        z = x.tensor.view(x.tensor.size(0), -1)
        out = self.classifier(z)

        out = out * self.class_scales.unsqueeze(0)

        return out, z


if __name__ == "__main__":
    model = ESCNNSteerable(ESCNNConfig())
    x = torch.randn(1, 3, 64, 64)
    out, z = model(x)
    print(out.shape, z.shape)
