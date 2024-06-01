# preprocessing.py
import cv2
import os

def apply_filters(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        median_filtered = cv2.medianBlur(image, 5)
        bilateral_filtered = cv2.bilateralFilter(median_filtered, 9, 75, 75)

        cv2.imwrite(os.path.join(output_dir, img_name), bilateral_filtered)

if __name__ == "__main__":
    apply_filters('data/augmented_images', 'data/filtered_images')
