# src/data_augmentation.py
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from src.cycle_gan_model import Generator, load_generator_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augment_data(input_dir, output_dir, model_path):
    model = load_generator_model(model_path)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            augmented_image = model(image)
        
        augmented_image = augmented_image.squeeze().cpu()
        save_path = os.path.join(output_dir, img_name)
        transforms.ToPILImage()(augmented_image).save(save_path)

if __name__ == "__main__":
    augment_data('data/ISIC_2019_Training_Input', 'data/augmented_images', 'models/cyclegan_model.pth')
