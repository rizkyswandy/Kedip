# Deployment Guide: Zero-to-Training Pipeline

## Perfect for GPU Server Deployment! 🚀

This guide shows you how to go from **nothing to training** on a GPU server with zero manual file transfers.

---

## Quick Start (GPU Server)

### 1. Clone & Setup

```bash
# SSH into your GPU server
ssh your-gpu-server

# Clone the repo
git clone https://github.com/yourusername/BlinkDetection.git
cd BlinkDetection

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Option A: Test with Dummy Data (Fastest)

```bash
# Create dummy data and start training immediately
uv run python train.py --config configs/dev_test.yaml --create-dummy

# This will:
# ✓ Create fake datasets for testing
# ✓ Start training right away
# ✓ No downloads needed
# ✓ Perfect for testing the pipeline
```

**Use this for**: Testing the pipeline, debugging, development

### 3. Option B: Auto-Download Real Datasets

```bash
# Download datasets automatically then train
uv run python train.py --config configs/production.yaml --download-datasets

# This will:
# ✓ Check which datasets are missing
# ✓ Download available datasets automatically
# ✓ Show instructions for manual datasets
# ✓ Start training when ready
```

**Use this for**: Production training with real data

### 4. Option C: Download Datasets First

```bash
# Download datasets separately
uv run python scripts/download_datasets.py --create-dummy

# Then train
uv run python train.py --config configs/production.yaml
```

---

## Dataset Download Options

### List Available Datasets

```bash
uv run python scripts/download_datasets.py --list
```

Output:
```
Available datasets:

  CEW             - Closed Eyes in the Wild (CEW) (~2GB)
  EyeBlink8       - EyeBlink8 Dataset (~500MB) [Manual]
  ZJU             - ZJU Eyeblink Database (~1GB) [Manual]
  Talkingface     - Talkingface Video Dataset (~3GB) [Manual]
  RT-GENE         - RT-GENE Dataset (~5GB) [Manual]
  RT-BENE         - RT-BENE Dataset (~10GB) [Manual]
```

### Download All Datasets

```bash
# Download all available datasets
uv run python scripts/download_datasets.py

# Download specific datasets
uv run python scripts/download_datasets.py --datasets CEW EyeBlink8

# Create dummy data for all datasets
uv run python scripts/download_datasets.py --create-dummy
```

### Manual Datasets

Some datasets require manual download due to access restrictions:

1. **EyeBlink8**: Visit https://www.blinkingmatters.com/research
2. **ZJU**: Visit http://www.cs.zju.edu.cn/research/eyeblink
3. **RT-GENE**: Visit https://github.com/Tobias-Fischer/rt_gene
4. **RT-BENE**: Visit https://www.tobiasfischer.info/rt-bene/

After downloading, place in `data/datasets/<DATASET_NAME>/`

---

## Complete Training Pipeline

### Development (Mac M2 or Testing)

```bash
# Test with dummy data
uv run python train.py --config configs/dev_test.yaml --create-dummy

# Monitor training
tensorboard --logdir experiments/
```

Runs in: ~2 hours

### Production (GPU Server)

```bash
# Full training with auto-download
uv run python train.py --config configs/production.yaml --download-datasets

# Or create dummy first (for testing pipeline)
uv run python train.py --config configs/production.yaml --create-dummy

# Monitor remotely (on GPU server)
tensorboard --logdir experiments/ --host 0.0.0.0 --port 6006

# On your local machine
ssh -L 6006:localhost:6006 your-gpu-server
# Open: http://localhost:6006
```

Runs in: 24-48 hours

---

## GitHub → GPU Server Workflow

### Step 1: Push to GitHub

```bash
# On your local machine
git add .
git commit -m "Add BlinkTransformer training pipeline"
git push origin main
```

### Step 2: Deploy to GPU Server

```bash
# SSH to GPU server
ssh your-gpu-server

# Clone repo
git clone https://github.com/yourusername/BlinkDetection.git
cd BlinkDetection

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### Step 3: Start Training

Choose your approach:

```bash
# A. Test with dummy data first
uv run python train.py --config configs/production.yaml --create-dummy

# B. Download real datasets and train
uv run python train.py --config configs/production.yaml --download-datasets

# C. Download datasets separately
uv run python scripts/download_datasets.py --create-dummy
uv run python train.py --config configs/production.yaml
```

### Step 4: Monitor Training

```bash
# Start TensorBoard on GPU server
tensorboard --logdir experiments/ --host 0.0.0.0 --port 6006

# On local machine, forward port
ssh -L 6006:localhost:6006 your-gpu-server

# Open browser: http://localhost:6006
```

### Step 5: Download Trained Model

```bash
# On local machine
scp -r your-gpu-server:BlinkDetection/experiments ./
scp -r your-gpu-server:BlinkDetection/checkpoints ./
```

---

## Training Commands Cheat Sheet

```bash
# Test with dummy data (fastest)
uv run python train.py --config configs/dev_test.yaml --create-dummy

# Download datasets automatically
uv run python train.py --config configs/production.yaml --download-datasets

# Train with existing datasets
uv run python train.py --config configs/production.yaml

# Resume from checkpoint
uv run python train.py --config configs/production.yaml \
    --resume checkpoints/best_model.pth

# Evaluation only
uv run python train.py --config configs/production.yaml \
    --eval-only --resume checkpoints/best_model.pth

# Monitor training
tensorboard --logdir experiments/
```

---

## Dataset Download Commands

```bash
# List available datasets
uv run python scripts/download_datasets.py --list

# Download all datasets
uv run python scripts/download_datasets.py

# Download specific datasets
uv run python scripts/download_datasets.py --datasets CEW EyeBlink8

# Create dummy data (for testing)
uv run python scripts/download_datasets.py --create-dummy

# Download to custom directory
uv run python scripts/download_datasets.py --output-dir /data/blink
```

---

## Configuration

### For Testing (Quick Iteration)

Edit `configs/dev_test.yaml`:

```yaml
experiment:
  device: mps  # or cuda or cpu

data:
  datasets:
    - name: CEW
      path: data/datasets/CEW
      weight: 1.0

training:
  batch_size: 4
  epochs: 10
```

### For Production (Full Training)

Edit `configs/production.yaml`:

```yaml
experiment:
  device: cuda

data:
  datasets:
    - name: CEW
      path: data/datasets/CEW
      weight: 1.0
    - name: EyeBlink8
      path: data/datasets/EyeBlink8
      weight: 1.0
    # Add more datasets...

training:
  batch_size: 16
  epochs: 100
  use_amp: true  # Mixed precision
```

---

## Troubleshooting

### "Missing datasets" Error

```bash
# Solution 1: Create dummy data
uv run python train.py --config configs/dev_test.yaml --create-dummy

# Solution 2: Download datasets
uv run python scripts/download_datasets.py --create-dummy

# Solution 3: Auto-download during training
uv run python train.py --config configs/dev_test.yaml --download-datasets
```

### "CUDA out of memory"

```bash
# Reduce batch size in config
# training:
#   batch_size: 8  # Instead of 16
```

### "No module named 'scripts'"

```bash
# Make sure you're in the project root
cd /path/to/BlinkDetection
```

### Port Already in Use (TensorBoard)

```bash
# Use different port
tensorboard --logdir experiments/ --port 6007
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Train Model

on:
  push:
    branches: [ main ]

jobs:
  train:
    runs-on: self-hosted  # Your GPU server
    steps:
      - uses: actions/checkout@v2

      - name: Setup uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: uv sync

      - name: Train with dummy data
        run: uv run python train.py --config configs/dev_test.yaml --create-dummy

      - name: Upload model
        uses: actions/upload-artifact@v2
        with:
          name: trained-model
          path: experiments/*/checkpoints/best_model.pth
```

---

## Best Practices

1. **Start with Dummy Data**: Test the pipeline with `--create-dummy` first
2. **Use Screen/Tmux**: For long training sessions
   ```bash
   screen -S training
   uv run python train.py --config configs/production.yaml --download-datasets
   # Ctrl+A, D to detach
   screen -r training  # To reattach
   ```

3. **Monitor GPU Usage**:
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Save Checkpoints Regularly**: Already configured in `configs/`

5. **Use Git for Model Versioning**:
   ```bash
   git tag v1.0.0-model
   git push --tags
   ```

---

## Production Checklist

- [ ] Clone repo on GPU server
- [ ] Install uv and dependencies
- [ ] Test with dummy data first
- [ ] Download/prepare real datasets
- [ ] Configure training parameters
- [ ] Start training with monitoring
- [ ] Set up automatic checkpointing
- [ ] Configure TensorBoard access
- [ ] Plan model download strategy

---

## Summary

**Zero file transfers needed!** Just:

1. Push code to GitHub
2. Clone on GPU server
3. Run: `uv run python train.py --config configs/production.yaml --create-dummy`
4. Start training immediately

**That's it!** No manual dataset downloads, no file copying, fully automated. 🎉

---

## Quick Reference

| Command | Purpose |
|---------|---------|
| `--create-dummy` | Create fake data for testing |
| `--download-datasets` | Auto-download real datasets |
| `--resume <path>` | Resume from checkpoint |
| `--eval-only` | Only run evaluation |
| `--list` | List available datasets |

---

**Next**: Read [SETUP_COMPLETE.md](SETUP_COMPLETE.md) for full feature documentation.
