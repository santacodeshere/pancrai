"""
PancrAI — Loss Functions
Combined Dice Loss + Binary Cross Entropy for segmentation training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for binary segmentation.

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)
    Loss = 1 - Dice

    Args:
        smooth: Smoothing factor to prevent division by zero.
        sigmoid: Apply sigmoid to logits before computing Dice.
    """

    def __init__(self, smooth: float = 1e-5, sigmoid: bool = True):
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            preds = torch.sigmoid(logits)
        else:
            preds = logits

        # Flatten spatial dimensions
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()

        intersection = (preds_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (
            preds_flat.sum() + targets_flat.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in segmentation.
    Focuses training on hard examples (misclassified pixels).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Weighting factor for positive class (default 0.25).
        gamma: Focusing parameter (default 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class CombinedDiceBCELoss(nn.Module):
    """
    Combined Dice + Binary Cross Entropy loss.

    Total_Loss = dice_weight * DiceLoss + bce_weight * BCELoss

    Default: 0.5 * Dice + 0.5 * BCE

    Args:
        dice_weight: Weight for Dice component (default 0.5).
        bce_weight: Weight for BCE component (default 0.5).
        pos_weight: Positive class weight for BCE (to handle imbalance).
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5,
                 pos_weight: float = 3.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=1e-5, sigmoid=True)
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to same device as logits
        pos_w = self.pos_weight.to(logits.device)

        dice = self.dice_loss(logits, targets)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), pos_weight=pos_w
        )
        return self.dice_weight * dice + self.bce_weight * bce


class TverskyLoss(nn.Module):
    """
    Tversky Loss — generalization of Dice that penalizes
    false negatives more than false positives (useful for
    small, hard-to-detect tumors).

    TL = TP / (TP + alpha*FP + beta*FN)
    Loss = 1 - TL

    With alpha=0.3, beta=0.7 → penalizes FN more (recall-focused).
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = torch.sigmoid(logits)
        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1).float()

        tp = (preds_flat * targets_flat).sum()
        fp = (preds_flat * (1 - targets_flat)).sum()
        fn = ((1 - preds_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss — applies focal penalty to Tversky.
    Excellent for small pancreatic tumor regions.

    FTL = (1 - TL)^gamma
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7,
                 gamma: float = 0.75, smooth: float = 1e-5):
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        tl = self.tversky(logits, targets)
        return tl ** self.gamma
