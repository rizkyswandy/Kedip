"""
Dataset loaders for blink detection.

Supports multiple public blink detection datasets:
- CEW (Closed Eyes in the Wild)
- EyeBlink8
- ZJU
- Talkingface
- RN
- RT-GENE
- RT-BENE
"""

import csv
import json
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .preprocessing import FeatureExtractor


class BlinkDataset(Dataset):
    """
    Generic blink detection dataset.

    Expected directory structure:
        dataset_root/
            videos/
                video001.mp4
                video002.mp4
                ...
            annotations/
                video001.json
                video002.json
                ...

    Annotation format (JSON):
        {
            "video_name": "video001.mp4",
            "fps": 30,
            "total_frames": 300,
            "blinks": [
                {
                    "start_frame": 45,
                    "end_frame": 52,
                    "duration_ms": 233
                },
                ...
            ],
            "frame_labels": [0, 0, 0, ..., 1, 1, 1, ..., 0]  # Optional
        }
    """

    def __init__(
        self,
        dataset_root: str,
        sequence_length: int = 16,
        stride: int = 8,
        split: str = 'train',
        transform=None,
        feature_extractor: FeatureExtractor | None = None
    ):
        """
        Args:
            dataset_root: Root directory of dataset
            sequence_length: Number of frames per sequence
            stride: Stride between sequences
            split: 'train', 'val', or 'test'
            transform: Optional transforms
            feature_extractor: Feature extractor instance
        """
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        self.transform = transform

        # Initialize feature extractor
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
            self.own_extractor = True
        else:
            self.feature_extractor = feature_extractor
            self.own_extractor = False

        # Load dataset
        self.sequences = self._load_sequences()

        print(f"Loaded {len(self.sequences)} sequences from {dataset_root} ({split})")

    def _load_sequences(self) -> list[dict]:
        """
        Load and prepare sequences from dataset.

        Returns:
            List of sequence metadata
        """
        sequences: list[dict] = []

        videos_dir = self.dataset_root / 'videos'
        annotations_dir = self.dataset_root / 'annotations'

        if not videos_dir.exists():
            print(f"Warning: Videos directory not found: {videos_dir}")
            return sequences

        # Get split file if exists
        split_file = self.dataset_root / f'{self.split}.txt'
        if split_file.exists():
            with open(split_file) as f:
                video_names = [line.strip() for line in f]
        else:
            # Use all videos
            video_names = [v.stem for v in videos_dir.glob('*.mp4')]

        # Process each video
        for video_name in video_names:
            video_path = videos_dir / f'{video_name}.mp4'
            annotation_path = annotations_dir / f'{video_name}.json'

            if not video_path.exists():
                continue

            # Load annotation if exists
            if annotation_path.exists():
                with open(annotation_path) as f:
                    annotation = json.load(f)
            else:
                # Create dummy annotation
                cap = cv2.VideoCapture(str(video_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                annotation = {
                    'video_name': video_name,
                    'total_frames': total_frames,
                    'blinks': [],
                    'frame_labels': [0] * total_frames
                }

            # Create sequences from this video
            video_sequences = self._create_sequences(
                str(video_path),
                annotation
            )
            sequences.extend(video_sequences)

        return sequences

    def _create_sequences(
        self,
        video_path: str,
        annotation: dict
    ) -> list[dict]:
        """
        Create overlapping sequences from a video.

        Args:
            video_path: Path to video file
            annotation: Video annotation

        Returns:
            List of sequence metadata
        """
        sequences = []
        total_frames = annotation['total_frames']
        frame_labels = annotation.get('frame_labels', [0] * total_frames)

        # Ensure frame_labels is complete
        if len(frame_labels) < total_frames:
            frame_labels.extend([0] * (total_frames - len(frame_labels)))

        # Create sequences with stride
        for start_idx in range(0, total_frames - self.sequence_length + 1, self.stride):
            end_idx = start_idx + self.sequence_length

            # Get labels for this sequence
            seq_labels = frame_labels[start_idx:end_idx]

            # Sequence has blink if any frame is labeled as blink
            has_blink = any(label == 1 for label in seq_labels)

            sequences.append({
                'video_path': video_path,
                'start_frame': start_idx,
                'end_frame': end_idx,
                'frame_labels': seq_labels,
                'has_blink': has_blink
            })

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a sequence.

        Returns:
            Dictionary with:
                - 'rgb': Eye patches (T, 3, H, W)
                - 'landmarks': Landmark features (T, 146)
                - 'pose': Head pose angles (T, 3)
                - 'presence': Binary label (1,) - has blink in sequence
                - 'state': Frame-level labels (T,) - per-frame eye state
        """
        seq_info = self.sequences[idx]

        # Load video frames
        frames = self._load_frames(
            seq_info['video_path'],
            seq_info['start_frame'],
            seq_info['end_frame']
        )

        # Extract features
        features = self.feature_extractor.extract_sequence(frames)

        if features is None:
            # Return dummy data if extraction fails
            return self._get_dummy_sample()

        # Apply transforms if any
        if self.transform:
            features = self.transform(features)

        # Add labels
        features['presence'] = torch.tensor(
            [float(seq_info['has_blink'])],
            dtype=torch.float32
        )
        features['state'] = torch.tensor(
            seq_info['frame_labels'],
            dtype=torch.float32
        )

        return features

    def _load_frames(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int
    ) -> list[np.ndarray]:
        """
        Load frames from video.

        Args:
            video_path: Path to video
            start_frame: Start frame index
            end_frame: End frame index (exclusive)

        Returns:
            List of frames (each H, W, 3)
        """
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frames: list[np.ndarray] = []
        for _ in range(end_frame - start_frame):
            ret, frame = cap.read()
            if not ret:
                # Repeat last frame if video ends early
                if frames:
                    frame = frames[-1].copy()
                else:
                    # Create black frame
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

            frames.append(frame)

        cap.release()
        return frames

    def _get_dummy_sample(self) -> dict[str, torch.Tensor]:
        """Create dummy sample when feature extraction fails."""
        return {
            'rgb': torch.zeros(self.sequence_length, 3, 64, 64),
            'landmarks': torch.zeros(self.sequence_length, 146),
            'pose': torch.zeros(self.sequence_length, 3),
            'presence': torch.zeros(1),
            'state': torch.zeros(self.sequence_length)
        }

    def __del__(self):
        """Cleanup."""
        if self.own_extractor:
            self.feature_extractor.close()


class RTBENEDataset(Dataset):
    """
    RT-BENE dataset loader for image-based blink detection.

    Expected directory structure:
        dataset_root/
            rt_bene_subjects.csv          # Master file
            s000_blink_labels.csv         # Blink labels per subject
            s000_noglasses/natural/left/  # Left eye images
            s000_noglasses/natural/right/ # Right eye images
            ...
    """

    def __init__(
        self,
        dataset_root: str,
        sequence_length: int = 16,
        stride: int = 8,
        split: str = 'train',
        transform=None,
        feature_extractor: FeatureExtractor | None = None
    ):
        """
        Args:
            dataset_root: Root directory of RT-BENE dataset
            sequence_length: Number of frames per sequence
            stride: Stride between sequences
            split: 'train' or 'val'
            transform: Optional transforms
            feature_extractor: Feature extractor instance
        """
        self.dataset_root = Path(dataset_root)
        self.sequence_length = sequence_length
        self.stride = stride
        self.split = split
        self.transform = transform

        # Initialize feature extractor
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
            self.own_extractor = True
        else:
            self.feature_extractor = feature_extractor
            self.own_extractor = False

        # Load dataset
        self.sequences = self._load_sequences()

        print(f"Loaded {len(self.sequences)} sequences from {dataset_root} ({split})")

    def _load_sequences(self) -> list[dict]:
        """Load and prepare sequences from RT-BENE dataset."""
        sequences: list[dict] = []

        # Read master CSV
        master_csv = self.dataset_root / 'rt_bene_subjects.csv'
        if not master_csv.exists():
            print(f"Warning: Master CSV not found: {master_csv}")
            return sequences

        # Parse master CSV to get subjects
        # CSV format: index,file,left,right,split,fold (no header row)
        with open(master_csv) as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue

                # Parse columns: index,file,left,right,split,fold
                labels_file = row[1]  # e.g., 's000_blink_labels.csv'
                left_dir = row[2]     # e.g., 's000_noglasses/natural/left/'
                right_dir = row[3]    # e.g., 's000_noglasses/natural/right/'
                subject_split = row[4]  # 'training' or 'validation'

                # Map split names
                if self.split == 'train' and subject_split != 'training':
                    continue
                if self.split == 'val' and subject_split != 'validation':
                    continue

                # Load this subject's sequences
                subject_sequences = self._load_subject(
                    labels_file,
                    left_dir,
                    right_dir
                )
                sequences.extend(subject_sequences)

        return sequences

    def _load_subject(
        self,
        labels_file: str,
        left_dir: str,
        right_dir: str
    ) -> list[dict]:
        """Load sequences from a single subject."""
        sequences: list[dict] = []

        # Read blink labels CSV
        labels_path = self.dataset_root / labels_file
        if not labels_path.exists():
            return sequences

        # Read labels: filename, label (0.0=open, 1.0=closed, 0.5=uncertain)
        # CSV format: name,label (no header row in some files)
        labels_data = []
        with open(labels_path) as f:
            reader = csv.reader(f)
            for row_idx, row in enumerate(reader):
                if len(row) < 2:
                    continue

                # Skip header if present (check if first row contains 'name' or 'label')
                if row_idx == 0 and (row[0].lower() == 'name' or row[1].lower() == 'label'):
                    continue

                try:
                    filename = row[0]
                    label = float(row[1])
                    labels_data.append({
                        'filename': filename,
                        'label': 1 if label >= 0.5 else 0  # Treat uncertain as closed
                    })
                except (ValueError, IndexError):
                    # Skip malformed rows
                    continue

        if len(labels_data) < self.sequence_length:
            return sequences

        # Get image directories
        left_path = self.dataset_root / left_dir
        right_path = self.dataset_root / right_dir

        if not left_path.exists() or not right_path.exists():
            return sequences

        # Create sequences with stride
        for start_idx in range(0, len(labels_data) - self.sequence_length + 1, self.stride):
            end_idx = start_idx + self.sequence_length

            # Get labels for this sequence
            seq_labels = [labels_data[i]['label'] for i in range(start_idx, end_idx)]
            has_blink = any(label == 1 for label in seq_labels)

            # Store sequence info
            sequences.append({
                'left_dir': str(left_path),
                'right_dir': str(right_path),
                'filenames': [labels_data[i]['filename'] for i in range(start_idx, end_idx)],
                'frame_labels': seq_labels,
                'has_blink': has_blink
            })

        return sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a sequence."""
        seq_info = self.sequences[idx]

        # Load images
        frames = self._load_images(
            seq_info['left_dir'],
            seq_info['right_dir'],
            seq_info['filenames']
        )

        # Extract features
        features = self.feature_extractor.extract_sequence(frames)

        if features is None:
            # Return dummy data if extraction fails
            return self._get_dummy_sample()

        # Apply transforms if any
        if self.transform:
            features = self.transform(features)

        # Add labels
        features['presence'] = torch.tensor(
            [float(seq_info['has_blink'])],
            dtype=torch.float32
        )
        features['state'] = torch.tensor(
            seq_info['frame_labels'],
            dtype=torch.float32
        )

        return features

    def _load_images(
        self,
        left_dir: str,
        right_dir: str,
        filenames: list[str]
    ) -> list[np.ndarray]:
        """
        Load eye images and combine them into frames.

        For RT-BENE, we have separate left/right eye images.
        We'll create a combined image by placing them side-by-side.
        """
        frames = []
        left_path = Path(left_dir)
        right_path = Path(right_dir)

        for filename in filenames:
            # Load left and right eye images
            left_img_path = left_path / filename
            right_img_path = right_path / filename

            if left_img_path.exists() and right_img_path.exists():
                left_img = cv2.imread(str(left_img_path))
                right_img = cv2.imread(str(right_img_path))

                if left_img is not None and right_img is not None:
                    # Combine side-by-side: [left | right]
                    combined = np.hstack([left_img, right_img])
                    frames.append(combined)
                    continue

            # If loading fails, create black frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((64, 128, 3), dtype=np.uint8))

        return frames

    def _get_dummy_sample(self) -> dict[str, torch.Tensor]:
        """Create dummy sample when feature extraction fails."""
        return {
            'rgb': torch.zeros(self.sequence_length, 3, 64, 64),
            'landmarks': torch.zeros(self.sequence_length, 146),
            'pose': torch.zeros(self.sequence_length, 3),
            'presence': torch.zeros(1),
            'state': torch.zeros(self.sequence_length)
        }

    def __del__(self):
        """Cleanup."""
        if self.own_extractor:
            self.feature_extractor.close()


class MultiDatasetLoader:
    """
    Loader for multiple datasets with weighted sampling.
    """

    def __init__(
        self,
        datasets_config: list[dict],
        sequence_length: int = 16,
        stride: int = 8,
        split: str = 'train',
        batch_size: int = 8,
        num_workers: int = 4,
        shuffle: bool = True
    ):
        """
        Args:
            datasets_config: List of dataset configurations
                [{'name': 'CEW', 'path': '...', 'weight': 1.0}, ...]
            sequence_length: Frames per sequence
            stride: Stride between sequences
            split: 'train', 'val', or 'test'
            batch_size: Batch size
            num_workers: Number of data loading workers
            shuffle: Shuffle data
        """
        self.datasets: list[Union[BlinkDataset, RTBENEDataset]] = []
        self.weights: list[float] = []

        # Create shared feature extractor
        feature_extractor = FeatureExtractor()

        # Load all datasets
        for config in datasets_config:
            dataset_path = config['path']
            dataset_name = config.get('name', '')

            if not Path(dataset_path).exists():
                print(f"Warning: Dataset not found: {dataset_path}")
                continue

            # Check if this is RT-BENE dataset (has master CSV file)
            is_rtbene = (Path(dataset_path) / 'rt_bene_subjects.csv').exists()

            dataset: Union[BlinkDataset, RTBENEDataset]
            if is_rtbene or dataset_name == 'RT-BENE':
                dataset = RTBENEDataset(
                    dataset_root=dataset_path,
                    sequence_length=sequence_length,
                    stride=stride,
                    split=split,
                    feature_extractor=feature_extractor
                )
            else:
                dataset = BlinkDataset(
                    dataset_root=dataset_path,
                    sequence_length=sequence_length,
                    stride=stride,
                    split=split,
                    feature_extractor=feature_extractor
                )

            self.datasets.append(dataset)
            self.weights.append(config.get('weight', 1.0))

        # Concatenate datasets
        self.combined_dataset: ConcatDataset = ConcatDataset(self.datasets)

        # Create data loader
        self.loader = DataLoader(
            self.combined_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


def create_dataloaders(
    config: dict,
    splits: list[str] = ['train', 'val']
) -> dict[str, MultiDatasetLoader]:
    """
    Create data loaders from configuration.

    Args:
        config: Configuration dictionary
        splits: List of splits to create

    Returns:
        Dictionary of data loaders
    """
    dataloaders = {}

    for split in splits:
        is_train = split == 'train'

        loader = MultiDatasetLoader(
            datasets_config=config['data']['datasets'],
            sequence_length=config['data']['sequence_length'],
            stride=config['data'].get('stride', 8),
            split=split,
            batch_size=config['training']['batch_size'] if is_train else config['validation']['batch_size'],
            num_workers=config['experiment'].get('num_workers', 4),
            shuffle=is_train
        )

        dataloaders[split] = loader

    return dataloaders


if __name__ == '__main__':
    """Test dataset loading."""

    print("Testing BlinkDataset...")

    # Create dummy dataset
    dataset_root = Path('data/datasets/test_dataset')
    dataset_root.mkdir(parents=True, exist_ok=True)

    videos_dir = dataset_root / 'videos'
    annotations_dir = dataset_root / 'annotations'
    videos_dir.mkdir(exist_ok=True)
    annotations_dir.mkdir(exist_ok=True)

    # Create a dummy annotation
    dummy_annotation = {
        'video_name': 'test_video',
        'fps': 30,
        'total_frames': 100,
        'blinks': [
            {'start_frame': 20, 'end_frame': 25},
            {'start_frame': 60, 'end_frame': 65}
        ],
        'frame_labels': [0] * 20 + [1] * 5 + [0] * 35 + [1] * 5 + [0] * 35
    }

    with open(annotations_dir / 'test_video.json', 'w') as f:
        json.dump(dummy_annotation, f)

    print(f"Created dummy dataset at {dataset_root}")
    print("Note: Add actual video files to test fully")

    print("\nDataset test setup complete! ✓")
    print("To test with real data:")
    print("  1. Add video files to data/datasets/YOUR_DATASET/videos/")
    print("  2. Add annotations to data/datasets/YOUR_DATASET/annotations/")
    print("  3. Run: python -c 'from data.datasets import BlinkDataset; ds = BlinkDataset(\"data/datasets/YOUR_DATASET\"); print(len(ds))'")
