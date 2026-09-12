# Hybrid Glaucoma Detection Model

A PyTorch-based deep-learning pipeline for classifying OCT and fundus images as **Normal** or **Glaucoma Suspect**. The project supports transfer learning with ResNet, DenseNet, or a ResNet-50/DenseNet-121 ensemble and evaluates models with stratified cross-validation.

## Features

- Reads image labels from the supplied Excel patient record.
- Loads images from either an extracted dataset directory or a ZIP archive.
- Converts the source labels into two classes:
  - `0`: Normal (`no`)
  - `1`: Glaucoma Suspect (`yes` or `suspect`)
- Uses stratified 5-fold cross-validation by default.
- Handles class imbalance with a weighted sampler.
- Applies training-time image augmentation and ImageNet normalization.
- Saves fold metrics, ROC/confusion-matrix plots, probability distributions, Grad-CAM-style visualizations, and the best checkpoint.
- Provides a command-line script for single-image prediction.

## Architecture

```text
Excel patient record + JPG images
              |
              v
       src/dataset.py
   label parsing and image matching
              |
              v
      train/validation transforms
              |
              v
       StratifiedKFold (5 folds)
              |
              v
  ResNet-50 + DenseNet-121 ensemble
       (or a configured single model)
              |
              v
     Cross-validation evaluation
              |
              v
 outputs/checkpoints + outputs/plots
```

The default ensemble averages the logits from a pretrained ResNet-50 and DenseNet-121. Each backbone has a replacement classifier with dropout and a two-class output layer. Pretrained weights are downloaded by `torchvision` the first time a model is created, so internet access is required for the initial run unless the weights are already cached.

## Data Flow

1. `main.py` reads `config/config.yaml`.
2. `src/dataset.py` searches the configured data path for an `.xlsx` file.
3. The first worksheet is read without assuming a header row. Rows after the first row are processed.
4. Label columns are read from Excel columns `1, 3, 5, 7`; the corresponding answer columns are used for voting.
5. The majority label is converted to binary form: `yes`/`suspect` becomes `1`, and `no` becomes `0`.
6. Image filenames from the records are matched against `.jpg` files by basename. Whitespace and apostrophes are tolerated during matching.
7. Images are split with stratified cross-validation. Training images receive random flips, rotation, and color jitter; validation images receive resize and normalization only.
8. The best model state for each fold is selected by validation ROC-AUC. Out-of-fold metrics and visualizations are generated after all folds finish.

## Repository Structure

```text
glucoma_detection_model/
├── config/
│   └── config.yaml                 # Dataset, model, and training settings
├── data/
│   └── Data on OCT and Fundus Images/
│       ├── Patient Record-final.xlsx
│       ├── P 1/                    # Patient image folders
│       ├── P 2/
│       └── ...
├── outputs/
│   ├── checkpoints/                # Saved PyTorch model weights
│   └── plots/                      # Metrics and visualization outputs
├── src/
│   ├── dataset.py                  # Dataset, labels, transforms, image loading
│   ├── models.py                   # ResNet, DenseNet, and ensemble definitions
│   ├── predict.py                  # Single-image inference CLI
│   ├── train.py                    # Cross-validation and training loop
│   └── utils.py                    # Grad-CAM and plotting utilities
├── explore_data.py                 # Excel and sample-image inspection script
├── main.py                         # Main training entry point
├── requirements.txt                # Python dependencies
└── README.md
```

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer recommended
- A CUDA-capable GPU is recommended for training, but the pipeline falls back to CPU automatically when CUDA is unavailable.
- Approximately 10 GB or more of free disk space is recommended for Python packages, cached pretrained weights, model checkpoints, and generated outputs.

## Setup From a Git Clone

Replace `<repository-url>` with the URL of this repository.

```powershell
git clone <repository-url>
cd glucoma_detection_model

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current PowerShell session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For a CUDA-enabled PyTorch installation, follow the current installation command for the installed NVIDIA driver from the [official PyTorch website](https://pytorch.org/get-started/locally/), then install the remaining packages from `requirements.txt`.

Verify the environment:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

## Dataset Setup

Place the dataset at exactly this path relative to the repository root:

```text
data/Data on OCT and Fundus Images/
```

The directory must contain:

```text
data/Data on OCT and Fundus Images/
├── Patient Record-final.xlsx
├── P 1/
├── P 2/
└── ...
```

The loader also accepts a ZIP file when `data.file_path` points to that file. The archive must contain at least one `.xlsx` label file and `.jpg` image files. The current configuration uses the extracted directory.

Run the data inspection script before training:

```powershell
python explore_data.py
```

This prints Excel metadata, displays the first sample image when the expected sample path exists, and helps confirm that the dataset path and file names are correct.

## Configuration

Edit [config/config.yaml](config/config.yaml) from the repository root:

```yaml
project:
  name: "Glaucoma_Detection_Pipeline"
  seed: 42
  output_dir: "outputs"

data:
  file_path: "data/Data on OCT and Fundus Images"
  img_size: 224
  num_classes: 2

training:
  model_name: "ensemble"  # resnet, densenet, or ensemble
  n_folds: 5
  epochs: 25
  batch_size: 8
  learning_rate: 0.0001
  weight_decay: 0.001
  device: "cuda"            # cuda or cpu
```

Important settings:

| Setting | Description |
| --- | --- |
| `data.file_path` | Dataset directory or ZIP path, relative to the repository root |
| `data.img_size` | Square image size used by the transforms |
| `training.model_name` | `resnet18`, `resnet34`, `resnet50`, `resnet`, `densenet`, or `ensemble` |
| `training.n_folds` | Number of stratified cross-validation folds |
| `training.epochs` | Training epochs per fold |
| `training.batch_size` | Batch size for training and validation |
| `training.device` | Requested device; CPU is selected automatically when CUDA is unavailable |
| `project.output_dir` | Root directory for checkpoints and plots |

## Train the Model

Run this command from the repository root:

```powershell
python main.py
```

The training process prints fold progress and the final classification report. With the default configuration it runs five folds for 25 epochs each, so runtime depends heavily on the GPU and dataset size.

The best final checkpoint is saved as:

```text
outputs/checkpoints/best_ensemble_model.pth
```

Generated plots include:

```text
outputs/plots/dl_fold_summary.png
outputs/plots/dl_roc_cm.png
outputs/plots/dl_prob_dist.png
outputs/plots/dl_gradcam.png
```

To run a smaller CPU smoke test, temporarily change the configuration:

```yaml
training:
  model_name: "resnet18"
  n_folds: 2
  epochs: 1
  batch_size: 2
  device: "cpu"
```

The corresponding checkpoint name will be `best_resnet18_model.pth`.

## Predict a Single Image

After training, use the checkpoint that matches `training.model_name` in the configuration:

```powershell
python -m src.predict `
  --image "data/Data on OCT and Fundus Images/P 1/109151_20150910_080317_B-scan_R_001.jpg" `
  --model "outputs/checkpoints/best_ensemble_model.pth"
```

The command prints the predicted class, glaucoma probability, and confidence. By default it also writes a Grad-CAM overlay to:

```text
outputs/plots/latest_prediction_heatmap.png
```

The model architecture in `config/config.yaml` must match the checkpoint architecture. For example, an ensemble checkpoint must be used with `model_name: ensemble`; a DenseNet checkpoint must be used with `model_name: densenet`.

## Useful Commands

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Inspect the dataset
python explore_data.py

# Train with the current configuration
python main.py

# Run inference on one image
python -m src.predict --image "path\to\image.jpg" --model "outputs\checkpoints\best_ensemble_model.pth"

# Check installed packages
python -m pip list
```

## Troubleshooting

### `FileNotFoundError: No .xlsx label file found`

Confirm that `data.file_path` points to the dataset directory or ZIP archive and that the directory contains the patient Excel record.

### No images are loaded

Confirm that the Excel filenames correspond to `.jpg` files in the dataset. The current loader searches for JPG files and matches by filename basename; PNG files are not included.

### CUDA is unavailable

Install a PyTorch build compatible with the local NVIDIA driver, or set `training.device` to `cpu`. The code automatically falls back to CPU when CUDA is not available.

### Model weights cannot be downloaded

The ResNet and DenseNet backbones use torchvision pretrained weights. Run once with internet access, or use a local torchvision cache before running training in an offline environment.

### Prediction checkpoint loading fails

Make sure the checkpoint was produced by the same model architecture configured in `config/config.yaml` and that the checkpoint path points to a `.pth` file.

## Notes

- This project is a research/development pipeline and is not a medical diagnostic device.
- Results depend on dataset quality, label quality, class balance, and the chosen configuration.
- Keep patient data private and follow the dataset's license, consent, and institutional requirements.