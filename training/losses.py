"""
Loss functions for blink detection.

Multi-task loss:
    - Presence loss: Binary classification (has blink in sequence)
    - State loss: Frame-level classification (eye state per frame)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BlinkLoss(nn.Module):
    """
    Multi-task loss for blink detection.

    Combines:
        1. Presence loss: BCE for sequence-level blink presence
        2. State loss: BCE for frame-level eye state
    """

    def __init__(
        self,
        presence_weight: float = 1.0,
        state_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        use_focal: bool = False
    ):
        """
        Args:
            presence_weight: Weight for presence loss
            state_weight: Weight for state loss
            focal_alpha: Alpha parameter for focal loss
            focal_gamma: Gamma parameter for focal loss
            use_focal: Use focal loss instead of BCE
        """
        super().__init__()

        self.presence_weight = presence_weight
        self.state_weight = state_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.use_focal = use_focal

    def forward(
        self,
        pred_presence: torch.Tensor,
        pred_state: torch.Tensor,
        target_presence: torch.Tensor,
        target_state: torch.Tensor
    ) -> tuple:
        """
        Compute multi-task loss.

        Args:
            pred_presence: Predicted presence (B, 1)
            pred_state: Predicted state (B, T)
            target_presence: Target presence (B, 1)
            target_state: Target state (B, T)

        Returns:
            (total_loss, presence_loss, state_loss)
        """
        # Presence loss (sequence-level)
        if self.use_focal:
            presence_loss = self.focal_loss(pred_presence, target_presence)
        else:
            presence_loss = F.binary_cross_entropy(
                pred_presence,
                target_presence,
                reduction='mean'
            )

        # State loss (frame-level)
        if self.use_focal:
            state_loss = self.focal_loss(pred_state, target_state)
        else:
            state_loss = F.binary_cross_entropy(
                pred_state,
                target_state,
                reduction='mean'
            )

        # Combined loss
        total_loss = (
            self.presence_weight * presence_loss +
            self.state_weight * state_loss
        )

        return total_loss, presence_loss, state_loss

    def focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.

        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

        Args:
            pred: Predictions
            target: Targets

        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')

        # Compute p_t
        p_t = pred * target + (1 - pred) * (1 - target)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.focal_gamma

        # Compute alpha weight
        alpha_t = self.focal_alpha * target + (1 - self.focal_alpha) * (1 - target)

        # Focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        return focal_loss.mean()


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency loss to ensure smooth predictions.
    Penalizes rapid changes in predictions.
    """

    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(self, pred_state: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal consistency loss.

        Args:
            pred_state: Predicted state (B, T)

        Returns:
            Consistency loss
        """
        # Compute differences between consecutive frames
        diff = pred_state[:, 1:] - pred_state[:, :-1]

        # L1 loss on differences
        consistency_loss = torch.abs(diff).mean()

        return self.weight * consistency_loss


class CombinedLoss(nn.Module):
    """
    Combined loss with temporal consistency.
    """

    def __init__(
        self,
        presence_weight: float = 1.0,
        state_weight: float = 1.0,
        consistency_weight: float = 0.1,
        use_focal: bool = False
    ):
        super().__init__()

        self.blink_loss = BlinkLoss(
            presence_weight=presence_weight,
            state_weight=state_weight,
            use_focal=use_focal
        )

        self.consistency_loss = TemporalConsistencyLoss(
            weight=consistency_weight
        )

    def forward(
        self,
        pred_presence: torch.Tensor,
        pred_state: torch.Tensor,
        target_presence: torch.Tensor,
        target_state: torch.Tensor
    ) -> tuple:
        """
        Compute combined loss.

        Returns:
            (total_loss, presence_loss, state_loss, consistency_loss)
        """
        # Main losses
        total_loss, presence_loss, state_loss = self.blink_loss(
            pred_presence,
            pred_state,
            target_presence,
            target_state
        )

        # Temporal consistency
        consistency = self.consistency_loss(pred_state)

        # Add consistency to total
        total_loss = total_loss + consistency

        return total_loss, presence_loss, state_loss, consistency


if __name__ == '__main__':
    """Test loss functions."""

    print("Testing loss functions...")

    # Create dummy predictions and targets
    B, T = 4, 16

    pred_presence = torch.rand(B, 1)
    pred_state = torch.rand(B, T)

    target_presence = torch.randint(0, 2, (B, 1)).float()
    target_state = torch.randint(0, 2, (B, T)).float()

    # Test BlinkLoss
    print("\n1. Testing BlinkLoss...")
    loss_fn = BlinkLoss(presence_weight=1.0, state_weight=1.0)
    total_loss, presence_loss, state_loss = loss_fn(
        pred_presence,
        pred_state,
        target_presence,
        target_state
    )

    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Presence loss: {presence_loss.item():.4f}")
    print(f"  State loss: {state_loss.item():.4f}")

    # Test with focal loss
    print("\n2. Testing BlinkLoss with Focal Loss...")
    focal_loss_fn = BlinkLoss(
        presence_weight=1.0,
        state_weight=1.0,
        use_focal=True
    )
    total_loss, presence_loss, state_loss = focal_loss_fn(
        pred_presence,
        pred_state,
        target_presence,
        target_state
    )

    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Presence loss: {presence_loss.item():.4f}")
    print(f"  State loss: {state_loss.item():.4f}")

    # Test TemporalConsistencyLoss
    print("\n3. Testing TemporalConsistencyLoss...")
    consistency_fn = TemporalConsistencyLoss(weight=0.1)
    consistency_loss = consistency_fn(pred_state)
    print(f"  Consistency loss: {consistency_loss.item():.4f}")

    # Test CombinedLoss
    print("\n4. Testing CombinedLoss...")
    combined_fn = CombinedLoss(
        presence_weight=1.0,
        state_weight=1.0,
        consistency_weight=0.1
    )
    total_loss, presence_loss, state_loss, consistency_loss = combined_fn(
        pred_presence,
        pred_state,
        target_presence,
        target_state
    )

    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Presence loss: {presence_loss.item():.4f}")
    print(f"  State loss: {state_loss.item():.4f}")
    print(f"  Consistency loss: {consistency_loss.item():.4f}")

    # Test backward pass
    print("\n5. Testing backward pass...")
    total_loss.backward()
    print("  Backward pass successful!")

    print("\nLoss functions test passed! ✓")
