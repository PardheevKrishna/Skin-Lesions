# Skin Lesion Classification

This project aims to classify skin lesions using a combination of CycleGAN for data augmentation, UNet for segmentation, and VGG16 for classification.

## Project Structure

skin_lesion_classification/
│
├── data/
│ ├── ISIC_2019_Training_Input/ # Contains all the images
│ ├── ISIC_2019_Training_GroundTruth.csv # Ground truth labels
│ └── ISIC_2019_Training_Metadata.csv # Metadata
|
├── src/
│ ├── init.py
│ ├── data_augmentation.py # Script for CycleGAN data augmentation
│ ├── preprocessing.py # Script for median and bilateral filtering
│ ├── segmentation.py # Script for segmentation using UNet
│ ├── feature_extraction.py # Script for GLCM, LBP, SIFT feature extraction
│ └── classification.py # Script for classification using VGG16
│
├── models/
│ ├── cyclegan_model.pth # Pretrained CycleGAN model
│ ├── unet_model.pt # Pretrained UNet model
│ └── vgg16_model.h5 # Pretrained VGG16 model
│
├── requirements.txt # Required packages and dependencies
├── main.py # Main script to run the entire pipeline
└── README.md # Project documentation


## Setup

1. Install the required packages:
    pip install -r requirements.txt

2. Run the main script:
    python main.py

### Additional Notes
- Make sure you have the pretrained models (`cyclegan_model.pth`, `unet_model.pt`, and `vgg16_model.h5`) in the `models/` directory.
- Customize the training parameters as needed in the individual scripts.

This structure and code should provide a comprehensive pipeline for skin lesion classification, following the specified order of operations and using the best practices for each step.