---
license: cc-by-4.0
task_categories:
- image-segmentation
tags:
- medical
- polyp-segmentation
- colonoscopy
- gastrointestinal
size_categories:
- n<1K
pretty_name: CVC-ClinicDB
---

# CVC-ClinicDB: Colonoscopy Polyp Segmentation Dataset

## Dataset Description

CVC-ClinicDB is a colonoscopy polyp segmentation dataset containing 612 frames extracted from 29 colonoscopy sequences with corresponding ground truth segmentation masks.

- **Task**: Binary segmentation (polyp vs. background)
- **Modality**: Colonoscopy
- **Format**: PNG images (384x288 pixels) with binary masks
- **Splits**: Training, Validation, and Test sets

## Dataset Structure

```
CVC-ClinicDB/
├── train/
│   ├── images/  # Training images
│   └── masks/   # Training masks
├── validation/
│   ├── images/  # Validation images
│   └── masks/   # Validation masks
└── test/
    ├── images/  # Test images
    └── masks/   # Test masks
```

## Citation

If you use this dataset, please cite:

```bibtex
@article{bernal2015wm,
  title={WM-DOVA maps for accurate polyp highlighting in colonoscopy: Validation vs. saliency maps from physicians},
  author={Bernal, Jorge and S{'a}nchez, F Javier and Fern{'a}ndez-Esparrach, Gloria and Gil, Debora and Rodr{\'\i}guez, Cristina and Vilari{\~n}o, Fernando},
  journal={Computerized Medical Imaging and Graphics},
  volume={43},
  pages={99--111},
  year={2015},
  publisher={Elsevier}
}
```

**Dataset Split**: This train/validation/test split is created by the following study. Please find more details in:

```bibtex
@article{chang2024esfpnet,
  title={ESFPNet: Efficient Stage-Wise Feature Pyramid on Mix Transformer for Deep Learning-Based Cancer Analysis in Endoscopic Video},
  author={Chang, Qi and Ahmad, Danish and Toth, Jennifer and Bascom, Rebecca and Higgins, William E},
  journal={Journal of Imaging},
  volume={10},
  number={8},
  pages={191},
  year={2024},
  publisher={MDPI}
}
```

## Usage

```python
from datasets import load_dataset

# Load dataset
dataset = load_dataset("Angelou0516/CVC-ClinicDB")

# Access splits
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Access a sample
sample = train_data[0]
image = sample['file_name']  # Image
mask = sample['mask_file_name']  # Segmentation mask
```

## License

Please refer to the original CVC-ClinicDB dataset license and citation requirements.

## Links

- Original Dataset: https://polyp.grand-challenge.org/CVCClinicDB/
- Paper: https://www.sciencedirect.com/science/article/pii/S0895611115000567
