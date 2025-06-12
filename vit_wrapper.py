import os
import torch
from PIL import Image
from torchvision import transforms as T

from ViTs import ViTWithTransformerBlocks, get_vit_specific_config_224 # Import the custom Vision Transformer model and its config loader 
from evaluation_transformers import load_pretrained_model # Import pre-trained model loader function 

# Load and configure the ViT model 
def load_model(device):
    model_arch = "google/vit-base-patch16-224"
    model_config = get_vit_specific_config_224(model_arch)
    model_config['image_size'] = 224
    model_config['mean'] = (0.5, 0.5, 0.5)
    model_config['std'] = (0.5, 0.5, 0.5)

    vit_dir = model_arch.replace("/", "_")
    checkpoint = os.path.join("experiments", vit_dir, "best.ckpt")

    model = load_pretrained_model(checkpoint, model_config, device=device)
    return model, model_config

# Open and preprocess images given their path and transform pipeline
def preprocess_image(img_path, transform):
    image = Image.open(img_path).convert("RGB")
    return transform(image).unsqueeze(0)  

# Calculate cosine similarity between 2 images 
def cosine_similarity(model, tensor1, tensor2, device):
    model.eval()
    with torch.no_grad():
        tensor1 = tensor1.to(device)
        tensor2 = tensor2.to(device)

        embed1, _ = model(tensor1)
        embed2, _ = model(tensor2)

        embed1 = torch.nn.functional.normalize(embed1, dim=-1) # Normalize
        embed2 = torch.nn.functional.normalize(embed2, dim=-1)

        similarity = torch.nn.functional.cosine_similarity(embed1, embed2)
        return similarity.item()

# Main script (load models, transform images, preprocess them, and compute the cosine similarity)
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, config = load_model(device)

    transform = T.Compose([
        T.Resize((config['image_size'], config['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=config['mean'], std=config['std'])
    ])

# Image paths
    img1 = preprocess_image("sample_images/Aaron_Sorkin_0001.jpg.png", transform)
    img2 = preprocess_image("sample_images/Aaron_Guiel_0001.jpg.png", transform)

    score = cosine_similarity(model, img1, img2, device)
    print(f"Cosine similarity: {score:.4f}")
