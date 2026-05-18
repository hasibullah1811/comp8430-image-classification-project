# Fine-Grained Image Classification Pipeline using MobileNetV3

An elegant, modular, and lightweight computer vision pipeline built in PyTorch for fine-grained visual categorization (FGVC). This project is designed to dynamically adapt to any target object domain containing 20+ fine-grained classes while prioritizing ultra-low computational overhead for real-time edge deployment on physical robotic platforms.

---

## 🏗️ System Architecture

The codebase strictly decouples the **Model Training & Optimization Pipeline** from the **Onboard Robot Inference Engine**. This architectural isolation ensures that we can train complex models on high-performance machines and seamlessly deploy a lightweight inference layer with zero training overhead to a robot's onboard system.

A detailed system architectural overview mapping the complete data ingestion, processing, and hardware deployment flow can be found in the repository root as `diagram.png`.

### Repository Structure

` ` `text
comp8430-image-classification-project/
│
├── data/                    # Dataset directory (Excluded from Git tracking)
│   ├── train/               # Training image subsets split into class folders
│   └── val/                 # Validation image subsets split into class folders
│
├── src/                     # Core system source code
│   ├── __init__.py          # Package initialization
│   ├── dataset.py           # Custom FineGrainedImageDataset & transforms
│   ├── model.py             # MobileNetV3 backbone setup & transfer learning head
│   ├── train.py             # Modular PyTorch training loop & checkpoint management
│   └── inference.py         # Standalone real-time edge inference script
│
├── create_test_data.py      # Automated dummy dataset generator utility
├── diagram.png              # Excalidraw-mapped system architecture diagram
├── LICENSE                  # MIT License
└── requirements.txt         # Project runtime dependencies
` ` `

---

## 🛠️ Getting Started

### 1. Prerequisites & Installation
Ensure you have a Python environment configured (Python 3.8+ recommended). Clone this repository, navigate to the directory, and install the required dependencies:

` ` `bash
# Install the core software requirements
pip install -r requirements.txt
` ` `

### 2. Dataset Layout (`ImageFolder` Pattern)
The codebase uses a dynamic directory scanning approach. It adapts automatically to whatever object classes are established simply by reading the folder names under the target directories:

` ` `text
data/
├── train/
│   ├── class_name_1/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── class_name_2/
│       └── ...
└── val/
    ├── class_name_1/
    └── ...
` ` `

---

## Pipeline Verification (Smoke Testing)

To verify that the dataloaders, neural connections, matrix transformations, and weight-saving systems work flawlessly on your local machine, run the following three steps:

### Step A: Generate Synthetic Test Data
Run the automated dummy generator to construct a temporary 3-class mock dataset made of solid color blocks:

` ` `bash
python create_test_data.py
` ` `

### Step B: Execute the Training Pipeline
Run a 2-epoch training pass. The script will dynamically overwrite the default target configurations to match the 3 mock classes generated in Step A:

` ` `bash
python src/train.py --data_dir ./data --epochs 2 --batch_size 4 --lr 0.001
` ` `
*Upon completion, confirm that the optimized weight binary `best_mobilenet.pth` has been written into the `src/` directory.*

### Step C: Test Onboard Robot Inference
Verify the lightweight edge inference runtime by passing a single image through the standalone prediction script:

` ` `bash
python src/inference.py --image_path ./data/val/class_a/dummy_0.jpg --checkpoint src/best_mobilenet.pth
` ` `
The terminal will instantly output a clean, robot-friendly prediction array:
` ` `text
Using device: cpu
Predicted class: class_a
Confidence: 0.4192
` ` `

---

## Core Architectural Specifications

| Component Layer | Configuration & Details |
| :--- | :--- |
| **Model Backbone** | `torchvision.models.mobilenet_v3_small` (Pre-trained on ImageNet) |
| **Feature Layer State**| Frozen feature extraction backbone to retain low-level edge kernels |
| **Optimization Target**| Dynamic multi-class Linear head mapping directly to dataset configuration |
| **Loss & Optimizer** | CrossEntropyLoss with Adam Optimizer |
| **Input Tensor Shape** | 3x224x224 (Width/Height matched to MobileNet input constraints) |
| **Data Augmentations** | Random Horizontal Flips + Slight Geometric Rotations (+/- 10 degrees) |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
EOF