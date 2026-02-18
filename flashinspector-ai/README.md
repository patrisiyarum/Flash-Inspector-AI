# FlashInspector AI - Fire Safety Detection System

AI-powered fire safety inspection using YOLOv8 object detection. Detects fire extinguishers, smoke/fire, emergency exit signs, and safety violations in images and video.

## Quick Start

```bash
# 1. Clone and enter the project
cd flashinspector-ai

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your Roboflow API key
cp .env.template .env
# Edit .env and add your key (get one free at https://app.roboflow.com/settings/api)

# 5. Download datasets
python download_datasets.py

# 6. Train a model
python fire_safety_datasets/train_model.py

# 7. Run inference
python fire_safety_datasets/test_model.py path/to/image_or_video
```

## API Key Setup

The project uses the [Roboflow](https://roboflow.com) API to download datasets. You need a free API key.

**Option 1 - `.env` file (recommended):**
```bash
cp .env.template .env
# Edit .env: ROBOFLOW_API_KEY=your_key_here
```

**Option 2 - Environment variable:**
```bash
export ROBOFLOW_API_KEY="your_key_here"
```

**Option 3 - Interactive prompt:**
If no key is found, the download script will prompt you to enter it.

> **Security:** The `.env` file is in `.gitignore` and will never be committed to version control.

## Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| `fire_extinguisher` | Fire extinguisher detection | Roboflow |
| `fire_smoke` | Fire and smoke detection | Roboflow |
| `emergency_exit` | Emergency exit sign detection | Roboflow |
| `construction_safety` | Construction site safety (PPE, cones, etc.) | Roboflow |

### Download Commands

```bash
# Download all datasets
python download_datasets.py

# Download a specific dataset
python download_datasets.py --dataset fire_extinguisher

# List available datasets
python download_datasets.py --list
```

## Training

```bash
# Train with defaults (fire_extinguisher dataset, nano model)
python fire_safety_datasets/train_model.py

# Choose dataset and model size
python fire_safety_datasets/train_model.py --dataset fire_smoke --size medium

# Custom epochs and batch size
python fire_safety_datasets/train_model.py --epochs 50 --batch 32

# Train without exporting
python fire_safety_datasets/train_model.py --no-export

# Export an existing model only
python fire_safety_datasets/train_model.py --export-only --weights fire_safety_models/fire_extinguisher_nano/weights/best.pt
```

### Model Sizes

| Size | Params | Speed | Accuracy | Best For |
|------|--------|-------|----------|----------|
| `nano` | ~3.2M | Fastest | Good | Mobile / edge devices |
| `small` | ~11.2M | Fast | Better | Balanced performance |
| `medium` | ~25.9M | Moderate | High | Server-side inference |
| `large` | ~43.7M | Slower | Highest | Maximum accuracy |

### Train on Google Colab (no strong local GPU)

Use Colab’s free GPU by opening the notebook and running the cells in order:

1. **Open in Colab:** [Open `train_on_colab.ipynb` in Colab](https://colab.research.google.com/github/YOUR_USERNAME/fire/blob/main/flashinspector-ai/train_on_colab.ipynb) (replace `YOUR_USERNAME` with your GitHub username after pushing), or upload `flashinspector-ai/train_on_colab.ipynb` to Colab (File → Upload notebook).
2. **Runtime → Change runtime type → T4 GPU** (or better).
3. Run the cells: get code (clone or use uploaded folder), install deps, set Roboflow API key, download dataset, train, then save `best.pt` to Google Drive so you keep it after the session ends.

The notebook uses Colab-friendly defaults (e.g. `--batch 8`, `--epochs 50`) so training fits in a free session.

### Train on Kaggle (GPU)

Use **`train_on_kaggle.ipynb`** in this repo: open [Kaggle Notebooks](https://www.kaggle.com/code), create a new notebook, then **File → Import Notebook from URL** and use your repo URL (e.g. `https://github.com/patrisiyarum/fire/blob/main/flashinspector-ai/train_on_kaggle.ipynb`), or clone the repo and upload the notebook. Enable **Settings → Accelerator → GPU**, add your Roboflow API key as a **Secret** (`ROBOFLOW_API_KEY`), and run all cells. The trained `best.pt` is copied to `/kaggle/working/` so you can download it from the Output tab.

### Export Formats

After training, models are automatically exported to:
- **ONNX** - Cross-platform deployment
- **TFLite** - Android deployment
- **CoreML** - iOS deployment

## Inference / Testing

```bash
# Run on an image
python fire_safety_datasets/test_model.py photo.jpg

# Run on a video (processes every 10th frame by default)
python fire_safety_datasets/test_model.py video.mp4

# Custom confidence threshold
python fire_safety_datasets/test_model.py video.mp4 --conf 0.5

# Process every frame (slower)
python fire_safety_datasets/test_model.py video.mp4 --frame-skip 1

# Show real-time display (requires GUI)
python fire_safety_datasets/test_model.py video.mp4 --show

# Batch process a folder
python fire_safety_datasets/test_model.py ./test_images/ --batch

# Save detections as JSON
python fire_safety_datasets/test_model.py video.mp4 --save-json

# Use a specific model
python fire_safety_datasets/test_model.py photo.jpg --model fire_safety_models/fire_smoke_medium/weights/best.pt
```

Results are saved to `inference_results/`.

## Project Structure

```
flashinspector-ai/
├── .env.template              # Template for API keys
├── .gitignore                 # Ignore .env, datasets, models
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── train_on_colab.ipynb       # Train on Google Colab (GPU)
├── train_on_kaggle.ipynb      # Train on Kaggle (GPU)
├── download_datasets.py       # Dataset downloader
├── fire_safety_datasets/      # Datasets & scripts
│   ├── combined_config.yaml   # All dataset classes
│   ├── train_model.py         # Training script
│   ├── test_model.py          # Inference script
│   └── <dataset_name>/        # Downloaded datasets (gitignored)
├── fire_safety_models/        # Trained models (gitignored)
│   └── <dataset>_<size>/
│       └── weights/
│           ├── best.pt
│           ├── best.onnx
│           ├── best.tflite
│           └── best.mlmodel
└── inference_results/         # Inference output
```

## Troubleshooting

**"No module named 'ultralytics'"**
```bash
pip install -r requirements.txt
```

**"ROBOFLOW_API_KEY not found"**
```bash
cp .env.template .env
# Edit .env with your API key
```

**"Dataset not found" during training**
```bash
python download_datasets.py  # Download datasets first
```

**CUDA out of memory**
```bash
# Reduce batch size
python fire_safety_datasets/train_model.py --batch 8
# Or use a smaller model
python fire_safety_datasets/train_model.py --size nano
```

**No GPU available**
Training will automatically fall back to CPU. GPU is recommended for faster training but not required.

**Video inference is slow**
```bash
# Increase frame skip (process fewer frames)
python fire_safety_datasets/test_model.py video.mp4 --frame-skip 30
```

## License

Proprietary - FlashInspector Inc.
