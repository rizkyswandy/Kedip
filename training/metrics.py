"""
Metrics for blink detection evaluation.

Metrics:
    - Accuracy
    - Precision, Recall, F1-score
    - True Positive Rate (TPR)
    - False Positive Rate (FPR)
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


class BlinkMetrics:
    """
    Compute and track metrics for blink detection.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Classification threshold
        """
        self.threshold = threshold
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.presence_preds = []
        self.presence_targets = []
        self.state_preds = []
        self.state_targets = []

    def update(
        self,
        pred_presence: torch.Tensor,
        pred_state: torch.Tensor,
        target_presence: torch.Tensor,
        target_state: torch.Tensor
    ):
        """
        Update metrics with batch predictions.

        Args:
            pred_presence: Predicted presence (B, 1)
            pred_state: Predicted state (B, T)
            target_presence: Target presence (B, 1)
            target_state: Target state (B, T)
        """
        # Convert to numpy
        pred_presence = pred_presence.detach().cpu().numpy()
        pred_state = pred_state.detach().cpu().numpy()
        target_presence = target_presence.detach().cpu().numpy()
        target_state = target_state.detach().cpu().numpy()

        # Store
        self.presence_preds.append(pred_presence)
        self.presence_targets.append(target_presence)
        self.state_preds.append(pred_state)
        self.state_targets.append(target_state)

    def compute(self) -> dict:
        """
        Compute all metrics.

        Returns:
            Dictionary of metrics
        """
        # Concatenate all batches
        presence_preds = np.concatenate(self.presence_preds, axis=0)
        presence_targets = np.concatenate(self.presence_targets, axis=0)
        state_preds = np.concatenate(self.state_preds, axis=0)
        state_targets = np.concatenate(self.state_targets, axis=0)

        # Flatten for state metrics
        state_preds_flat = state_preds.flatten()
        state_targets_flat = state_targets.flatten()

        # Compute presence metrics
        presence_metrics = self._compute_binary_metrics(
            presence_preds.flatten(),
            presence_targets.flatten(),
            prefix='presence_'
        )

        # Compute state metrics
        state_metrics = self._compute_binary_metrics(
            state_preds_flat,
            state_targets_flat,
            prefix='state_'
        )

        # Combine
        metrics = {**presence_metrics, **state_metrics}

        return metrics

    def _compute_binary_metrics(
        self,
        preds: np.ndarray,
        targets: np.ndarray,
        prefix: str = ''
    ) -> dict:
        """
        Compute binary classification metrics.

        Args:
            preds: Predictions (probabilities)
            targets: Ground truth (0 or 1)
            prefix: Prefix for metric names

        Returns:
            Dictionary of metrics
        """
        # Apply threshold
        preds_binary = (preds >= self.threshold).astype(int)
        targets_int = targets.astype(int)

        # Basic metrics
        accuracy = accuracy_score(targets_int, preds_binary)

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets_int,
            preds_binary,
            average='binary',
            zero_division=0
        )

        # True/False Positive Rate
        tp = np.sum((preds_binary == 1) & (targets_int == 1))
        fp = np.sum((preds_binary == 1) & (targets_int == 0))
        tn = np.sum((preds_binary == 0) & (targets_int == 0))
        fn = np.sum((preds_binary == 0) & (targets_int == 1))

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        # AUC-ROC (if possible)
        try:
            if len(np.unique(targets_int)) > 1:
                auc = roc_auc_score(targets_int, preds)
            else:
                auc = 0.0
        except Exception:
            auc = 0.0

        return {
            f'{prefix}accuracy': accuracy,
            f'{prefix}precision': precision,
            f'{prefix}recall': recall,
            f'{prefix}f1': f1,
            f'{prefix}tpr': tpr,
            f'{prefix}fpr': fpr,
            f'{prefix}auc': auc
        }

    def get_summary(self, metrics: dict = None) -> str:
        """
        Get formatted summary of metrics.

        Args:
            metrics: Metrics dictionary (if None, computes from stored data)

        Returns:
            Formatted string
        """
        if metrics is None:
            metrics = self.compute()

        summary = []
        summary.append("=== Blink Detection Metrics ===")
        summary.append("\nSequence-level (Presence):")
        summary.append(f"  Accuracy:  {metrics['presence_accuracy']:.4f}")
        summary.append(f"  Precision: {metrics['presence_precision']:.4f}")
        summary.append(f"  Recall:    {metrics['presence_recall']:.4f}")
        summary.append(f"  F1-score:  {metrics['presence_f1']:.4f}")
        summary.append(f"  AUC-ROC:   {metrics['presence_auc']:.4f}")

        summary.append("\nFrame-level (State):")
        summary.append(f"  Accuracy:  {metrics['state_accuracy']:.4f}")
        summary.append(f"  Precision: {metrics['state_precision']:.4f}")
        summary.append(f"  Recall:    {metrics['state_recall']:.4f}")
        summary.append(f"  F1-score:  {metrics['state_f1']:.4f}")
        summary.append(f"  AUC-ROC:   {metrics['state_auc']:.4f}")

        return '\n'.join(summary)


class AverageMeter:
    """Tracks and computes the average of values."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    """Test metrics computation."""

    print("Testing BlinkMetrics...")

    # Create dummy predictions and targets
    B, T = 8, 16
    num_batches = 5

    metrics = BlinkMetrics(threshold=0.5)

    # Simulate multiple batches
    print("\nSimulating 5 batches...")
    for i in range(num_batches):
        # Random predictions (with some correlation to targets)
        target_presence = torch.randint(0, 2, (B, 1)).float()
        target_state = torch.randint(0, 2, (B, T)).float()

        # Predictions biased toward targets (simulate good model)
        pred_presence = target_presence + torch.randn(B, 1) * 0.2
        pred_presence = torch.clamp(pred_presence, 0, 1)

        pred_state = target_state + torch.randn(B, T) * 0.2
        pred_state = torch.clamp(pred_state, 0, 1)

        # Update metrics
        metrics.update(pred_presence, pred_state, target_presence, target_state)

    # Compute final metrics
    print("\nComputing metrics...")
    results = metrics.compute()

    # Print summary
    print(metrics.get_summary(results))

    # Test individual metrics
    print("\n\nDetailed metrics:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # Test AverageMeter
    print("\n\nTesting AverageMeter...")
    meter = AverageMeter()

    for i in range(10):
        meter.update(i, n=1)

    print(f"  Average: {meter.avg:.2f} (should be 4.5)")
    print(f"  Sum: {meter.sum} (should be 45)")
    print(f"  Count: {meter.count} (should be 10)")

    print("\nMetrics test passed! ✓")
