"""
Main training script for BlinkTransformer.

Usage:
    # Quick test on Mac M2
    python train.py --config configs/dev_test.yaml

    # Full production training on GPU
    python train.py --config configs/production.yaml

    # Resume from checkpoint
    python train.py --config configs/dev_test.yaml --resume checkpoints/best_model.pth
"""

import argparse
import random

import numpy as np
import torch
import yaml

from data.datasets import create_dataloaders
from models import create_model
from training import Trainer
from scripts.download_datasets import setup_datasets


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def setup_device(config: dict) -> str:
    """Setup compute device."""
    device_name = config['experiment']['device']

    if device_name == 'cuda':
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("CUDA not available, using CPU")

    elif device_name == 'mps':
        if torch.backends.mps.is_available():
            device = 'mps'
            print("Using MPS (Apple Silicon)")
        else:
            device = 'cpu'
            print("MPS not available, using CPU")

    else:
        device = 'cpu'
        print("Using CPU")

    return device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train BlinkTransformer')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--eval-only',
        action='store_true',
        help='Only run evaluation'
    )
    parser.add_argument(
        '--download-datasets',
        action='store_true',
        help='Download missing datasets automatically'
    )
    parser.add_argument(
        '--create-dummy',
        action='store_true',
        help='Create dummy data for testing without real datasets'
    )

    args = parser.parse_args()

    # Load config
    print(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Set seed
    set_seed(config['experiment']['seed'])
    print(f"Random seed: {config['experiment']['seed']}")

    # Setup device
    device = setup_device(config)

    # Setup datasets (auto-download if needed)
    print("\n" + "="*50)
    print("Dataset Setup")
    print("="*50)

    if args.create_dummy:
        print("Creating dummy data for testing...")
        datasets_ready = setup_datasets(config, create_dummy=True)
    elif args.download_datasets:
        print("Checking and downloading datasets...")
        datasets_ready = setup_datasets(config, create_dummy=False)
    else:
        # Just check if datasets exist
        from pathlib import Path
        datasets_config = config['data']['datasets']
        missing = []

        for ds in datasets_config:
            ds_path = Path(ds['path'])
            if not ds_path.exists() or not any(ds_path.iterdir()):
                missing.append(ds['name'])

        if missing:
            print(f"\n⚠️  Missing datasets: {', '.join(missing)}")
            print("\nTo download automatically:")
            print("  python train.py --config <config> --download-datasets")
            print("\nFor testing without real data:")
            print("  python train.py --config <config> --create-dummy")
            print("\nOr manually download datasets using:")
            print("  python scripts/download_datasets.py")
            return

        datasets_ready = True

    if not datasets_ready:
        print("\n❌ Dataset setup failed. Please check the errors above.")
        return

    print("✓ All datasets ready")

    # Create model
    print("\n" + "="*50)
    print("Model Creation")
    print("="*50)
    model = create_model(config['model'])
    print(f"Model parameters: {model.get_num_params():,}")

    # Create data loaders
    print("\n" + "="*50)
    print("Data Loaders")
    print("="*50)
    try:
        dataloaders = create_dataloaders(config, splits=['train', 'val'])
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']

        print(f"Training batches: {len(train_loader)}")
        print(f"Validation batches: {len(val_loader)}")

    except Exception as e:
        print(f"\nWarning: Could not load datasets: {e}")
        print("This is expected if you haven't downloaded the datasets yet.")
        print("\nTo train the model:")
        print("  1. Download datasets (see docs)")
        print("  2. Place them in data/datasets/")
        print("  3. Run this script again")
        print("\nFor now, you can test the model architecture:")
        print("  python models/blink_transformer.py")
        return

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        experiment_dir='experiments'
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Evaluate only
    if args.eval_only:
        print("\nRunning evaluation...")
        val_metrics = trainer.validate()

        print("\n" + "="*50)
        print("Evaluation Results")
        print("="*50)
        print(trainer.val_metrics.get_summary(val_metrics))
        print("="*50)

        return

    # Train
    print("\n" + "="*50)
    print(f"Starting Training: {config['experiment']['name']}")
    print("="*50 + "\n")

    trainer.train()

    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir}")
    print(f"Logs saved to: {trainer.log_dir}")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {trainer.log_dir}")
    print("\nTo test the model:")
    print(f"  python demo/webcam_demo.py --model {trainer.checkpoint_dir}/best_model.pth")


if __name__ == '__main__':
    main()
