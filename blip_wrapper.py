import os
import torch
from PIL import Image
from transformers import Blip2Model, Blip2Processor
from torch import nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Define device (GPU or CPU)

# BLIP2-based model wrapper
class BLIPforModel(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.feature_extractor = Blip2Model.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        self.feature_extractor = self.feature_extractor.vision_model
        self.processor = Blip2Processor.from_pretrained(model_id)

    def forward(self, image_path): # Load and format the image
        image = Image.open(image_path).convert("RGB")
        images = [image]
        pixels = self.processor(images=images, return_tensors="pt").pixel_values

        with torch.no_grad():
            image_features = self.feature_extractor(pixels.to(device))
            last_hidden_state = image_features.last_hidden_state
            image_feature = last_hidden_state[:, 0, :]  # Extract embedding of CLS token 

        return image_feature


# Calculate the cosine similarity
def cosine_similarity(model, img1, img2):
    model.eval()

    emb1 = model(img1).to(device)
    emb2 = model(img2).to(device)

    # Normalize
    emb1 = torch.nn.functional.normalize(emb1, dim=-1)
    emb2 = torch.nn.functional.normalize(emb2, dim=-1)

    similarity_score = torch.nn.functional.cosine_similarity(emb1, emb2)
    return similarity_score.item()


if __name__ == "__main__":
    model_name = "Salesforce/blip2-opt-2.7b"     # Model ID

    model = BLIPforModel(model_name)     # Load BLIP2 model
    model.eval()

    # Image paths
    img1 = "sample_images/Aaron_Sorkin_0001.jpg.png"
    img2 = "sample_images/Aaron_Guiel_0001.jpg.png"

    score = cosine_similarity(model, img1, img2)     # Compute similarity
    print(f"Cosine similarity between images: {score:.4f}")
