# src/segmentation.py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.unet_model import UNet  # Ensure correct import

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def save_image(image_tensor, path):
    image = transforms.ToPILImage()(image_tensor)
    image.save(path)

def segment_images(input_dir, output_dir, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = load_image(img_path)
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            segmented_image = model(image)

        save_image(segmented_image.cpu().squeeze(0), os.path.join(output_dir, img_name))

if __name__ == "__main__":
    segment_images('data/filtered_images', 'data/segmented_images', 'models/unet_model.pt')
