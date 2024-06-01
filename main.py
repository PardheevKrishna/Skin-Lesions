# main.py
import os
from src.data_augmentation import augment_data
from src.preprocessing import apply_filters
from src.segmentation import segment_images
from src.feature_extraction import extract_features
from src.classification import train_vgg16_model

def main():
    os.makedirs('data/augmented_images', exist_ok=True)
    os.makedirs('data/filtered_images', exist_ok=True)
    os.makedirs('data/segmented_images', exist_ok=True)

    augment_data('data/ISIC_2019_Training_Input', 'data/augmented_images', 'models/cyclegan_model.pth')
    apply_filters('data/augmented_images', 'data/filtered_images')
    segment_images('data/filtered_images', 'data/segmented_images', 'models/unet_model.pt')
    extract_features('data/segmented_images', 'data/features.npy')
    train_vgg16_model('data/features.npy', 'data/ISIC_2019_Training_GroundTruth.csv')

if __name__ == "__main__":
    main()
