# Skin Lesion Classification

This project aims to classify skin lesions using a combination of CycleGAN for data augmentation, UNet for segmentation, and VGG16 for classification.

## Project Structure

skin_lesion_classification/
│
├── data/
│   ├── ISIC_2019_Training_Input/
│   ├── ISIC_2019_Training_GroundTruth.csv
│   ├── ISIC_2019_Training_Metadata.csv
│   ├── augmented_images/
│   ├── segmented_images/
│   └── features.npy
│
├── notebooks/
│   ├── data_augmentation.ipynb
│   ├── preprocessing.ipynb
│   ├── segmentation.ipynb
│   ├── feature_extraction.ipynb
│   └── classification.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_augmentation.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   ├── classification.py
│   ├── cycle_gan_model.py
│   └── unet_model.py
│
├── models/
│   ├── cyclegan_model.pth
│   ├── unet_model.pt
│   └── vgg16_model.h5
│
├── requirements.txt
├── main.py
└── README.md



## Setup

1. Install the required packages:
    pip install -r requirements.txt

2. Run the main script:
    python main.py

### Additional Notes
- Make sure you have the pretrained models (`cyclegan_model.pth`, `unet_model.pt`, and `vgg16_model.h5`) in the `models/` directory.
- Customize the training parameters as needed in the individual scripts.

This structure and code should provide a comprehensive pipeline for skin lesion classification, following the specified order of operations and using the best practices for each step.
