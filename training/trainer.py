"""
Training loop for blink detection.

Features:
    - Mixed precision training
    - Gradient clipping
    - Learning rate scheduling
    - TensorBoard logging
    - Checkpoint saving
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .losses import BlinkLoss, CombinedLoss
from .metrics import AverageMeter, BlinkMetrics


class Trainer:
    """Training manager for blink detection."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: str = 'cuda',
        experiment_dir: str = 'experiments'
    ):
        """
        Args:
            model: BlinkTransformer model
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Configuration dictionary
            device: Device to train on
            experiment_dir: Directory for experiments
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Setup directories
        self.experiment_dir = Path(experiment_dir) / config['experiment']['name']
        self.checkpoint_dir = self.experiment_dir / 'checkpoints'
        self.log_dir = self.experiment_dir / 'logs'

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)

        # Setup loss
        self.criterion = self._setup_loss()

        # Setup optimizer
        self.optimizer = self._setup_optimizer()

        # Setup scheduler
        self.scheduler = self._setup_scheduler()

        # Setup mixed precision
        self.use_amp = config['training'].get('use_amp', False)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None

        # Metrics
        self.train_metrics = BlinkMetrics()
        self.val_metrics = BlinkMetrics()

        # Logging
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_f1 = 0.0

        print(f"Trainer initialized. Experiment: {self.experiment_dir}")

    def _setup_loss(self) -> nn.Module:
        """Setup loss function."""
        use_consistency = self.config['training'].get('use_consistency', False)

        if use_consistency:
            return CombinedLoss(
                presence_weight=self.config['training']['presence_weight'],
                state_weight=self.config['training']['state_weight'],
                consistency_weight=self.config['training'].get('consistency_weight', 0.1),
                use_focal=self.config['training'].get('use_focal', False)
            )
        else:
            return BlinkLoss(
                presence_weight=self.config['training']['presence_weight'],
                state_weight=self.config['training']['state_weight'],
                use_focal=self.config['training'].get('use_focal', False)
            )

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer."""
        optimizer_name = self.config['training']['optimizer'].lower()

        if optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        return optimizer

    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler | None:
        """Setup learning rate scheduler."""
        scheduler_name = self.config['training'].get('scheduler', 'cosine').lower()

        if scheduler_name == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        elif scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training'].get('scheduler_step', 30),
                gamma=0.1
            )
        elif scheduler_name == 'none':
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

        return scheduler

    def train_epoch(self) -> dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        loss_meter = AverageMeter()
        presence_loss_meter = AverageMeter()
        state_loss_meter = AverageMeter()

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            pose = batch['pose'].to(self.device)
            target_presence = batch['presence'].to(self.device)
            target_state = batch['state'].to(self.device)

            # Forward pass
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                pred_presence, pred_state = self.model(rgb, landmarks, pose)

                # Compute loss
                if isinstance(self.criterion, CombinedLoss):
                    loss, presence_loss, state_loss, _ = self.criterion(
                        pred_presence, pred_state,
                        target_presence, target_state
                    )
                else:
                    loss, presence_loss, state_loss = self.criterion(
                        pred_presence, pred_state,
                        target_presence, target_state
                    )

            # Backward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                # Gradient clipping
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )

                self.optimizer.step()

            # Update metrics
            loss_meter.update(loss.item(), rgb.size(0))
            presence_loss_meter.update(presence_loss.item(), rgb.size(0))
            state_loss_meter.update(state_loss.item(), rgb.size(0))

            self.train_metrics.update(
                pred_presence, pred_state,
                target_presence, target_state
            )

            # Logging
            if batch_idx % self.config['logging']['log_every'] == 0:
                pbar.set_postfix({
                    'loss': f'{loss_meter.avg:.4f}',
                    'p_loss': f'{presence_loss_meter.avg:.4f}',
                    's_loss': f'{state_loss_meter.avg:.4f}'
                })

                self.writer.add_scalar('train/loss', loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/presence_loss', presence_loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/state_loss', state_loss_meter.avg, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Compute epoch metrics
        metrics = self.train_metrics.compute()
        metrics['loss'] = loss_meter.avg
        metrics['presence_loss'] = presence_loss_meter.avg
        metrics['state_loss'] = state_loss_meter.avg

        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Validate model."""
        self.model.eval()
        self.val_metrics.reset()

        loss_meter = AverageMeter()
        presence_loss_meter = AverageMeter()
        state_loss_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc='Validation')

        for batch in pbar:
            # Move to device
            rgb = batch['rgb'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            pose = batch['pose'].to(self.device)
            target_presence = batch['presence'].to(self.device)
            target_state = batch['state'].to(self.device)

            # Forward pass
            pred_presence, pred_state = self.model(rgb, landmarks, pose)

            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, presence_loss, state_loss, _ = self.criterion(
                    pred_presence, pred_state,
                    target_presence, target_state
                )
            else:
                loss, presence_loss, state_loss = self.criterion(
                    pred_presence, pred_state,
                    target_presence, target_state
                )

            # Update metrics
            loss_meter.update(loss.item(), rgb.size(0))
            presence_loss_meter.update(presence_loss.item(), rgb.size(0))
            state_loss_meter.update(state_loss.item(), rgb.size(0))

            self.val_metrics.update(
                pred_presence, pred_state,
                target_presence, target_state
            )

            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}'
            })

        # Compute metrics
        metrics = self.val_metrics.compute()
        metrics['loss'] = loss_meter.avg
        metrics['presence_loss'] = presence_loss_meter.avg
        metrics['state_loss'] = state_loss_meter.avg

        return metrics

    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config['training']['epochs']} epochs...")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Experiment directory: {self.experiment_dir}\n")

        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch()

            # Validate
            if epoch % self.config['training']['eval_every'] == 0:
                val_metrics = self.validate()

                # Log validation metrics
                for key, value in val_metrics.items():
                    self.writer.add_scalar(f'val/{key}', value, epoch)

                # Print summary
                print(f"\nEpoch {epoch}/{self.config['training']['epochs']}")
                print(f"Train - Loss: {train_metrics['loss']:.4f}, "
                      f"F1: {train_metrics['presence_f1']:.4f}")
                print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"F1: {val_metrics['presence_f1']:.4f}")
                print(f"Time: {time.time() - start_time:.2f}s")

                # Save best model
                if val_metrics['presence_f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['presence_f1']
                    self.save_checkpoint('best_model.pth')
                    print(f"New best model! F1: {self.best_val_f1:.4f}")

            # Save periodic checkpoint
            if epoch % self.config['training']['save_every'] == 0:
                self.save_checkpoint(f'epoch_{epoch}.pth')

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

        print("\nTraining complete!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")

        # Save final model
        self.save_checkpoint('final_model.pth')

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'config': self.config
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_f1 = checkpoint['best_val_f1']

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.epoch}")


if __name__ == '__main__':
    """Test trainer setup."""
    print("Testing Trainer...")
    print("Note: Trainer requires data loaders and model to test fully")
    print("Run full training with: python train.py --config configs/dev_test.yaml")
    print("\nTrainer module created successfully! ✓")
