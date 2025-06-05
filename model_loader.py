import os
from ViTs  import ViTWithTransformerBlocks, get_vit_specific_config_224
import torch
from evaluation_transformers import load_pretrained_model
from torchvision import transforms as T
from PIL import Image

# List of ViT model names with input resolution of 224x224
vit_model_names_224 = [
    "google/vit-base-patch16-224",
    "google/vit-base-patch32-224-in21k",
    "google/vit-large-patch16-224",
    "google/vit-large-patch16-224-in21k",
    "google/vit-large-patch32-224-in21k",
    "google/vit-huge-patch14-224-in21k", ]

model_arch = vit_model_names_224[2]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# Load model config
model_config = get_vit_specific_config_224(model_arch)
model_config['image_size'] = int(model_arch.split("-")[3])
model_config['std'] = (0.5, 0.5, 0.5)
model_config['mean'] = (0.5, 0.5, 0.5)
    
vit_model = model_arch.replace("/", "_")  # Replace '/' with '_' for directory compatibility
checkpoint_path = os.path.join("./experiments", vit_model, "best.ckpt")



model = load_pretrained_model(checkpoint_path, model_config, device=device)
print(f"Model {model_arch} loaded successfully with config: {model_config}")


transformation_pipeline = T.Compose([
        T.Resize((model_config['image_size'], model_config['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=model_config['mean'], std=model_config['std'])
    ])


# Example usage of the transformation pipeline
img1_path = "sample_images/Aaron_Sorkin_0001.jpg.png"
img2_path = "sample_images/Aaron_Eckhart_0001.jpg.png"

img1 = Image.open(img1_path).convert("RGB")
img2 = Image.open(img2_path).convert("RGB")


img1_transformed = transformation_pipeline(img1)
img2_transformed = transformation_pipeline(img2)


feature1, _ = model(img1_transformed.unsqueeze(0).to(device))
feature2, _ = model(img2_transformed.unsqueeze(0).to(device))


print(f"Feature shape for {img1_path}: {feature1.shape}")
print(f"Feature shape for {img2_path}: {feature2.shape}")

# Normalize features
feature1 = torch.nn.functional.normalize(feature1, dim=-1)
feature2 = torch.nn.functional.normalize(feature2, dim=-1)


#Calculate cosine similarity
cosine_similarity = torch.nn.functional.cosine_similarity(feature1, feature2)
print(f"Cosine similarity between {img1_path} and {img2_path}: {cosine_similarity.item()}")











