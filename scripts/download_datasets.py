"""
Automatic dataset download script.

Downloads and prepares all required datasets for training.
No manual file transfer needed!

Usage:
    # Download all datasets
    python scripts/download_datasets.py

    # Download specific datasets
    python scripts/download_datasets.py --datasets CEW EyeBlink8

    # Specify output directory
    python scripts/download_datasets.py --output-dir data/datasets
"""

import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests
from tqdm import tqdm


# Dataset URLs and metadata
# Using Zenodo API for automated downloads
DATASETS = {
    'RT-BENE': {
        'description': 'RT-BENE Dataset (Blink Estimation)',
        'size': '~937MB',
        'zenodo_record': '3685316',
        'type': 'zenodo_multi_file',
        'files': {
            'master_csv': 'rt_bene_subjects.csv',
            'subjects': [f's{i:03d}' for i in range(17)],  # s000 to s016
        }
    }
}


def download_file(url: str, output_path: Path, desc: str = 'Downloading') -> bool:
    """
    Download file with progress bar.

    Args:
        url: Download URL
        output_path: Path to save file
        desc: Description for progress bar

    Returns:
        True if successful
    """
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f, tqdm(
            total=total_size,
            unit='B',
            unit_scale=True,
            desc=desc
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract zip file with progress bar.

    Args:
        zip_path: Path to zip file
        extract_to: Directory to extract to

    Returns:
        True if successful
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            members = zip_ref.namelist()
            for member in tqdm(members, desc='Extracting'):
                zip_ref.extract(member, extract_to)

        return True

    except Exception as e:
        print(f"Error extracting {zip_path}: {e}")
        return False


def extract_tar(tar_path: Path, extract_to: Path) -> bool:
    """
    Extract tar file with progress bar.

    Args:
        tar_path: Path to tar file
        extract_to: Directory to extract to

    Returns:
        True if successful
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)

        with tarfile.open(tar_path, 'r') as tar_ref:
            members = tar_ref.getmembers()
            for member in tqdm(members, desc=f'Extracting {tar_path.name}'):
                tar_ref.extract(member, extract_to)

        return True

    except Exception as e:
        print(f"Error extracting {tar_path}: {e}")
        return False


def create_dummy_annotations(dataset_path: Path, num_videos: int = 5):
    """
    Create dummy annotations and videos for testing without real data.

    Args:
        dataset_path: Path to dataset directory
        num_videos: Number of dummy videos to create annotations for
    """
    annotations_dir = dataset_path / 'annotations'
    videos_dir = dataset_path / 'videos'
    annotations_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating {num_videos} dummy annotations and videos for testing...")

    for i in range(num_videos):
        # Create annotation
        annotation = {
            'video_name': f'video{i:03d}',
            'fps': 30,
            'total_frames': 300,
            'blinks': [
                {'start_frame': 50, 'end_frame': 55},
                {'start_frame': 150, 'end_frame': 157},
                {'start_frame': 250, 'end_frame': 254}
            ],
            'frame_labels': [0] * 50 + [1] * 5 + [0] * 95 + [1] * 7 + [0] * 93 + [1] * 4 + [0] * 46
        }

        with open(annotations_dir / f'video{i:03d}.json', 'w') as f:
            json.dump(annotation, f, indent=2)

        # Create dummy video file
        video_path = videos_dir / f'video{i:03d}.mp4'
        if not video_path.exists():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore[attr-defined]
            out = cv2.VideoWriter(str(video_path), fourcc, 30, (640, 480))

            for frame_idx in range(300):
                # Create a black frame with text
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(
                    frame,
                    f'{dataset_path.name} - Video {i:03d} - Frame {frame_idx}',
                    (50, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
                out.write(frame)

            out.release()

    print(f"✓ Created dummy data at {dataset_path}")


def download_rt_bene(output_dir: Path, dataset_path: Path) -> bool:
    """
    Download RT-BENE dataset from Zenodo using their API.

    Args:
        output_dir: Temporary download directory
        dataset_path: Final dataset directory

    Returns:
        True if successful
    """
    zenodo_record = '3685316'
    base_url = f'https://zenodo.org/api/records/{zenodo_record}/files'

    dataset_path.mkdir(parents=True, exist_ok=True)

    # Download master CSV
    print("Downloading master CSV file...")
    master_csv_url = f'{base_url}/rt_bene_subjects.csv/content'
    if not download_file(master_csv_url, dataset_path / 'rt_bene_subjects.csv', 'rt_bene_subjects.csv'):
        return False

    # Download all subject files (s000 to s016)
    print("\nDownloading subject files...")
    for i in range(17):
        subject_id = f's{i:03d}'
        print(f"\nProcessing {subject_id}...")

        # Download blink labels CSV
        csv_filename = f'{subject_id}_blink_labels.csv'
        csv_url = f'{base_url}/{csv_filename}/content'
        if not download_file(csv_url, dataset_path / csv_filename, csv_filename):
            print(f"Warning: Failed to download {csv_filename}")
            continue

        # Download eye images TAR
        tar_filename = f'{subject_id}_noglasses_eyes.tar'
        tar_url = f'{base_url}/{tar_filename}/content'
        tar_path = output_dir / tar_filename

        if not download_file(tar_url, tar_path, tar_filename):
            print(f"Warning: Failed to download {tar_filename}")
            continue

        # Extract TAR file
        print(f"Extracting {tar_filename}...")
        if not extract_tar(tar_path, dataset_path):
            print(f"Warning: Failed to extract {tar_filename}")
            tar_path.unlink(missing_ok=True)
            continue

        # Clean up TAR file
        tar_path.unlink()

    print(f"\n✓ RT-BENE downloaded and extracted to {dataset_path}")
    return True


def download_dataset(
    dataset_name: str,
    output_dir: Path,
    create_dummy: bool = False
) -> bool:
    """
    Download and prepare a single dataset.

    Args:
        dataset_name: Name of dataset
        output_dir: Output directory
        create_dummy: Create dummy data if download not available

    Returns:
        True if successful or skipped
    """
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        return False

    dataset_info = DATASETS[dataset_name]
    dataset_path = output_dir / dataset_name

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_info['description']}")
    print(f"Size: {dataset_info['size']}")
    print(f"{'='*60}")

    # Check if already exists
    if dataset_path.exists() and any(dataset_path.iterdir()):
        print(f"✓ {dataset_name} already exists at {dataset_path}")
        return True

    # Check if manual download required
    if dataset_info.get('manual', False):
        print(f"\n⚠️  Manual download required for {dataset_name}")
        print(f"Instructions: {dataset_info['instructions']}")
        print(f"URL: {dataset_info['url']}")

        if create_dummy:
            print("\nCreating dummy data for testing...")
            dataset_path.mkdir(parents=True, exist_ok=True)
            (dataset_path / 'videos').mkdir(exist_ok=True)
            create_dummy_annotations(dataset_path)
            print("✓ Dummy data created (for testing only)")
            return True
        else:
            print("\nTo create dummy data for testing, use: --create-dummy")
            return False

    # Handle Zenodo multi-file datasets
    dataset_type = dataset_info.get('type', 'single_file')

    if dataset_type == 'zenodo_multi_file':
        if dataset_name == 'RT-BENE':
            return download_rt_bene(output_dir, dataset_path)
        else:
            print(f"Error: Unknown Zenodo multi-file dataset: {dataset_name}")
            return False

    # Automatic download for single files
    url = dataset_info.get('url')
    if not url or not isinstance(url, str):
        print(f"Error: No download URL specified for {dataset_name}")
        return False

    print(f"Downloading from {url}...")

    # Download to temporary location
    temp_file = output_dir / f'{dataset_name}.zip'

    if not download_file(url, temp_file, f'Downloading {dataset_name}'):
        return False

    # Extract
    print(f"Extracting {dataset_name}...")
    if not extract_zip(temp_file, output_dir):
        return False

    # Move to correct location if needed
    extract_subdir = dataset_info.get('extract_subdir')
    if extract_subdir is not None and isinstance(extract_subdir, str):
        extracted_path = output_dir / extract_subdir
        if extracted_path.exists() and extracted_path != dataset_path:
            extracted_path.rename(dataset_path)

    # Clean up
    temp_file.unlink()

    print(f"✓ {dataset_name} downloaded and extracted to {dataset_path}")
    return True


def download_all_datasets(
    output_dir: Path,
    datasets: Optional[list[str]] = None,
    create_dummy: bool = False
) -> dict[str, bool]:
    """
    Download all or specified datasets.

    Args:
        output_dir: Output directory
        datasets: List of dataset names (None = all)
        create_dummy: Create dummy data for manual datasets

    Returns:
        Dictionary of dataset_name: success
    """
    if datasets is None:
        datasets = list(DATASETS.keys())

    results = {}

    print(f"\n{'='*60}")
    print(f"Downloading {len(datasets)} dataset(s)")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    for dataset_name in datasets:
        success = download_dataset(dataset_name, output_dir, create_dummy)
        results[dataset_name] = success

    # Print summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")

    for dataset_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {dataset_name}")

    successful = sum(results.values())
    print(f"\nSuccessful: {successful}/{len(results)}")

    return results


def setup_datasets(config: dict, create_dummy: bool = False) -> bool:
    """
    Setup datasets based on config file.
    Automatically called by training script.

    Args:
        config: Training configuration
        create_dummy: Create dummy data for testing

    Returns:
        True if all required datasets are ready
    """
    datasets_config = config['data']['datasets']
    output_dir = Path('data/datasets')

    print("\nChecking datasets...")

    missing_datasets = []

    for dataset_config in datasets_config:
        dataset_name = dataset_config['name']
        dataset_path = Path(dataset_config['path'])

        if not dataset_path.exists() or not any(dataset_path.iterdir()):
            missing_datasets.append(dataset_name)

    if not missing_datasets:
        print("✓ All datasets available")
        return True

    print(f"\n⚠️  Missing datasets: {', '.join(missing_datasets)}")

    if create_dummy:
        print("\nCreating dummy data for testing...")
        download_all_datasets(
            output_dir,
            datasets=missing_datasets,
            create_dummy=True
        )
        return True
    else:
        print("\nDownloading datasets...")
        results = download_all_datasets(
            output_dir,
            datasets=missing_datasets,
            create_dummy=False
        )

        all_successful = all(results.values())

        if not all_successful:
            print("\n⚠️  Some datasets require manual download.")
            print("For testing without real data, use: --create-dummy")
            return False

        return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Download blink detection datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all datasets
  python scripts/download_datasets.py

  # Download specific datasets
  python scripts/download_datasets.py --datasets CEW EyeBlink8

  # Create dummy data for testing
  python scripts/download_datasets.py --create-dummy

  # Specify output directory
  python scripts/download_datasets.py --output-dir /path/to/datasets
        """
    )

    parser.add_argument(
        '--datasets',
        nargs='+',
        choices=list(DATASETS.keys()),
        help='Datasets to download (default: all)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/datasets',
        help='Output directory (default: data/datasets)'
    )

    parser.add_argument(
        '--create-dummy',
        action='store_true',
        help='Create dummy data for manual datasets (for testing)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available datasets'
    )

    args = parser.parse_args()

    # List datasets
    if args.list:
        print("\nAvailable datasets:\n")
        for name, info in DATASETS.items():
            manual = " [Manual]" if info.get('manual', False) else ""
            print(f"  {name:15} - {info['description']} ({info['size']}){manual}")
        print()
        return

    # Download datasets
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_all_datasets(
        output_dir,
        datasets=args.datasets,
        create_dummy=args.create_dummy
    )


if __name__ == '__main__':
    main()
