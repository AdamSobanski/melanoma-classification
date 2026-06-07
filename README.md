# Melanoma Classification

Binary classification of skin lesions (benign vs. malignant melanoma)
using transfer learning on EfficientNet-B0.

## Stack
- PyTorch + TorchVision
- EfficientNet-B0 (pretrained on ImageNet)
- Dataset: Melanoma Skin Cancer Dataset (10k images)

## Results
| Epoch | Loss   | Accuracy |
|-------|--------|----------|
| 1     | 0.2260 | 91.12%   |
| 2     | 0.1784 | 93.03%   |
| 3     | 0.1635 | 93.61%   |

**Test accuracy: 92.00%**

## Getting Started

1. Create conda environment:
```bash
conda create -n melanoma python=3.10
conda activate melanoma
```

2. Install dependencies:
```bash
pip install torch torchvision pillow matplotlib scikit-learn
```

3. Download dataset from Kaggle:
https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images

4. Place data in:
data/melanoma_cancer_dataset/train/
data/melanoma_cancer_dataset/test/

5. Open `notebooks/train.ipynb` and run all cells.

## Project Structure
melanoma_segmentation/
├── data/
│   └── melanoma_cancer_dataset/
│       ├── train/
│       │   ├── benign/
│       │   └── malignant/
│       └── test/
│           ├── benign/
│           └── malignant/
├── models/                  # model weights (not in repo, create by yourself using train.ipynb)
├── notebooks/
│   └── train.ipynb          # training notebook (already in repo)
└── src/                     # source code (this is space for next steps: segmentation etc. now is empty)

## Next Steps
- Streamlit app for inference
- Skin lesion segmentation (U-Net)
- Integration with full-body imaging pipeline