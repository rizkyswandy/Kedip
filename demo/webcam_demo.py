"""
Webcam demo for real-time blink detection.

Usage:
    # With trained model
    python demo/webcam_demo.py --model checkpoints/best_model.pth --device mps

    # Process video file
    python demo/webcam_demo.py --model checkpoints/best_model.pth --video input.mp4 --output output.mp4

    # Different camera
    python demo/webcam_demo.py --model checkpoints/best_model.pth --camera 1
"""

import argparse
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import RealtimeBlinkDetector
from inference.realtime import VideoProcessor
from models import BlinkTransformer, create_model


def load_model(model_path: str, device: str) -> BlinkTransformer:
    """
    Load trained model from checkpoint.

    Args:
        model_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Loading model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create model from config
    if 'config' in checkpoint:
        model_config = checkpoint['config']['model']
    else:
        # Use default config
        model_config = {
            'backbone': 'mobilenetv3_small_100',
            'hidden_dim': 256,
            'num_heads': 8,
            'num_layers': 4,
            'dropout': 0.1,
            'use_cross_modal': True,
            'use_linear_attention': True
        }

    model = create_model(model_config)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully!")
    print(f"Parameters: {model.get_num_params():,}")

    return model


def setup_device(device_name: str) -> str:
    """Setup compute device."""
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


def run_webcam_demo(args):
    """Run webcam demo."""
    # Setup device
    device = setup_device(args.device)

    # Load model
    model = load_model(args.model, device)

    # Create detector
    print("\nInitializing detector...")
    detector = RealtimeBlinkDetector(
        model=model,
        device=device,
        sequence_length=args.sequence_length,
        threshold=args.threshold
    )

    # Create processor
    processor = VideoProcessor(detector, visualize=True)

    # Process input
    if args.video:
        print(f"\nProcessing video: {args.video}")
        results = processor.process_video(args.video, args.output)

        print("\n" + "="*50)
        print("Results:")
        print("="*50)
        print(f"Total frames: {results['total_frames']}")
        print(f"Total blinks: {results['total_blinks']}")
        print(f"Blink rate: {results['total_blinks'] / (results['total_frames'] / 30):.2f} blinks/sec")
        print(f"Average FPS: {results['avg_fps']:.1f}")

        if results['blink_timestamps']:
            print("\nBlink timestamps (seconds):")
            for i, ts in enumerate(results['blink_timestamps'][:10], 1):
                print(f"  {i}. {ts:.2f}s")
            if len(results['blink_timestamps']) > 10:
                print(f"  ... and {len(results['blink_timestamps']) - 10} more")

        if args.output:
            print(f"\nOutput saved to: {args.output}")

    else:
        print(f"\nStarting webcam (camera {args.camera})...")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Reset statistics")
        print()

        processor.process_webcam(args.camera)

    # Cleanup
    detector.close()
    print("\nDemo complete!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Real-time blink detection demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam demo
  python demo/webcam_demo.py --model checkpoints/best_model.pth

  # Process video
  python demo/webcam_demo.py --model checkpoints/best_model.pth --video input.mp4

  # Save output
  python demo/webcam_demo.py --model checkpoints/best_model.pth --video input.mp4 --output output.mp4

  # Use different device
  python demo/webcam_demo.py --model checkpoints/best_model.pth --device cpu
        """
    )

    # Required arguments
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    # Input source
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Path to input video file (if not specified, uses webcam)'
    )
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )

    # Output
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save output video'
    )

    # Model settings
    parser.add_argument(
        '--device',
        type=str,
        default='mps',
        choices=['cuda', 'mps', 'cpu'],
        help='Compute device (default: mps)'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=16,
        help='Number of frames to analyze (default: 16)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Classification threshold (default: 0.5)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("\nTo train a model:")
        print("  python train.py --config configs/dev_test.yaml")
        sys.exit(1)

    if args.video and not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Run demo
    try:
        run_webcam_demo(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
