# src/feature_extraction.py
import os
import numpy as np
from skimage.feature import local_binary_pattern
from skimage.feature.texture import graycomatrix, graycoprops
import cv2

def extract_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    glcm = greycomatrix(gray_image, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = greycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
    energy = greycoprops(glcm, 'energy')[0, 0]
    correlation = greycoprops(glcm, 'correlation')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation]

def extract_lbp_features(image, radius=3, n_points=24):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_sift_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)
    return descriptors

def extract_features(input_dir, output_file):
    feature_list = []
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        glcm_features = extract_glcm_features(image)
        lbp_features = extract_lbp_features(image)
        sift_features = extract_sift_features(image)
        features = np.hstack([glcm_features, lbp_features, sift_features.flatten()])
        feature_list.append(features)
    np.save(output_file, feature_list)

if __name__ == "__main__":
    extract_features('data/segmented_images', 'data/features.npy')
