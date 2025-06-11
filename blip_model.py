

from transformers import Blip2Model, Blip2Processor
import torch
from torch import nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "Salesforce/blip2-opt-2.7b"

class BLIPforFR(nn.Module):
    def __init__(self, model_id):
        super().__init__()

        self.feature_extractor = Blip2Model.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
        self.feature_extractor = self.feature_extractor.vision_model
        self.processor = Blip2Processor.from_pretrained(model_id)

    def forward(self, image_path):
        image = Image.open(image_path).convert("RGB")
        images = [image]
        pixel_values = self.processor(images=images, return_tensors="pt").pixel_values


        with torch.no_grad():
            image_features = self.feature_extractor(pixel_values.to(device))
            last_hidden_state = image_features.last_hidden_state.detach().cpu()
            image_feature = last_hidden_state[:, 0, :]

        return image_feature



model = BLIPforFR(model_id)
model.eval()



if __name__ == "__main__":
  features = model(image_path = "/content/images/000002.jpg")
  print(features.shape)
  


