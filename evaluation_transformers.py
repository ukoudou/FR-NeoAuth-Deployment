
#!/usr/bin/env python

import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# -------------------------------------------------------------------
# 1) Import your model utilities from ViTs
# -------------------------------------------------------------------
from ViTs import ViTWithTransformerBlocks, get_vit_specific_config_224


def fuse_features_with_norm(stacked_embeddings, stacked_norms):
    assert stacked_embeddings.ndim == 3  # (n_features_to_fuse, batch_size, channel)
    assert stacked_norms.ndim == 3       # (n_features_to_fuse, batch_size, 1)
    
    # Multiply each embedding by its corresponding norm
    pre_norm_embeddings = stacked_embeddings * stacked_norms
    
    # Sum across the n_features_to_fuse dimension
    fused = pre_norm_embeddings.sum(dim=0)

    # L2-normalize along the embedding (channel) dimension
    fused_norm = torch.norm(fused, dim=1, keepdim=True)
    fused = fused / (fused_norm + 1e-9)

    return fused, fused_norm



# -------------------------------------------------------------------
# 2) Define a function to load the pretrained model
# -------------------------------------------------------------------
def load_pretrained_model(model_path, model_config, device=None):
    """
    Load a pretrained model from a checkpoint file and return it in eval mode.
    """


    model = ViTWithTransformerBlocks(
        pretrained_model_name=model_config['pretrained_model_name'],
        hidden_size=model_config['hidden_size'],
        ff_hidden_size=model_config['ff_hidden_size'],
        num_heads=model_config['num_heads'],
        embedding_dim=512
    )
    
    checkpoint_data = torch.load(model_path, map_location=device)
    print("Loading model from checkpoint file:", model_path)

    state_dictionary = checkpoint_data['state_dict']
    # Remove "model." prefix if present in the keys
    state_dictionary = {key.replace("model.", ""): val for key, val in state_dictionary.items()}
    model.load_state_dict(state_dictionary, strict=False)

    if device is not None:
        model = model.to(device=device).eval()
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device=device).eval()
    return model


# -------------------------------------------------------------------
# 3) Create a custom Dataset for reading images + optional transforms
# -------------------------------------------------------------------
class ImageFeaturesDataset(Dataset):
    def __init__(self, df_file, root_dir, transform=None):
        """
        Args:
            txt_file (str): Path to a .txt file with one image filename per line
            root_dir (str): Directory where these images are located
            transform (callable, optional): Transform pipeline (e.g., Resize, ToTensor)
        """
        df = pd.read_csv(df_file, delimiter=",", skipinitialspace=True)
        self.image_files = df["image"].values.tolist()
        

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.root_dir, image_file)
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, image_file.split("/")[-1] 


# -------------------------------------------------------------------
# 4) Main script entry point
# -------------------------------------------------------------------
def main(model_arch, model_path, output_dir, device=None):
    # ----------------------------
    # A) Define your model/checkpoint paths and config
    # ----------------------------
        
    

    # Load model config
    model_config = get_vit_specific_config_224(model_arch)
    model_config['image_size'] = int(model_arch.split("-")[3])
    model_config['std'] = (0.5, 0.5, 0.5)
    model_config['mean'] = (0.5, 0.5, 0.5)



    # ----------------------------
    # B) Define transformation pipeline
    # ----------------------------
    transformation_pipeline = T.Compose([
        T.Resize((model_config['image_size'], model_config['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=model_config['mean'], std=model_config['std'])
    ])

    # ----------------------------
    # # C) Instantiate Dataset and DataLoader
    # # ----------------------------
    # df_file_path = "/user/sonymd/iprobe-sonymd/NeoAuth-BioCOP/BIOCOP_PREPROCESSED/dataframes/biocop_images.csv"
    # root_dir_path = "/user/sonymd/iprobe-sonymd/NeoAuth-BioCOP/BIOCOP_PREPROCESSED"
    
    # 
    root_dir_path = "/user/sonymd/iprobe-sonymd/Diffuser-Super-Resolution"
    df_file_path ="/user/sonymd/iprobe-sonymd/Diffuser-Super-Resolution/dataframes/all_upscaled_images.csv"

    dataset = ImageFeaturesDataset(
        df_file=df_file_path,
        root_dir=root_dir_path,
        transform=transformation_pipeline
    )

    # Adjust batch_size and num_workers as appropriate for your system
    dataloader = DataLoader(dataset,
                            batch_size=128,
                            shuffle=False,
                            num_workers=8,
                            pin_memory=True)

    # ----------------------------
    # D) Load the pretrained model
    # ----------------------------
    model = load_pretrained_model(model_path, model_config, device=device)
    model.eval()
    

    # ----------------------------
    # E) Extract features in batches
    # ----------------------------
    features = {}
    # output_path = "/user/sonymd/iprobe-sonymd/NeoAuth-BioCOP/BIOCOP_PREPROCESSED/train_image_features.pkl"
    output_path = os.path.join(output_dir, "features.pkl")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    

    with torch.no_grad():
        for images, batch_filenames in tqdm(dataloader, desc="Extracting Features", total=len(dataloader)):
            # Send images to device
            images = images.to(device)
            # Flip the images horizontally along W dimension (dims=[3])
            flipped_images = torch.flip(images, dims=[3]).to(device)

            # Forward pass
            ft, norms = model(images)  # (batch_size, 512), (batch_size, ?)
            flipped_ft, flipped_norms = model(flipped_images)

            # # Stack embeddings and norms along a new dimension (dim=0)
            stacked_embeddings = torch.stack([ft, flipped_ft], dim=0)
            stacked_norms = torch.stack([norms, flipped_norms], dim=0)
            
            # # Fuse the original and flipped embeddings
            fused_embeddings, fused_norms = fuse_features_with_norm(stacked_embeddings, stacked_norms)

            # Store each feature vector in a dict with key=filename
            for i, filename in enumerate(batch_filenames):
                features[filename] = fused_embeddings[i].cpu().numpy()

    # ----------------------------
    # F) Save features to a pickle file
    # ----------------------------
    with open(output_path, "wb") as f:
        pickle.dump(features, f)

    print(f"Saved {len(features)} feature vectors to\n {output_path}\n\n\n")


if __name__ == "__main__":
    google_vit_models = sorted([x  for x in os.listdir('./experiments') if x.startswith("google_vit")])
    # output_dir = "/user/sonymd/iprobe-sonymd/NeoAuth-BioCOP/BIOCOP_PREPROCESSED/extracted_features"
    output_dir = "/user/sonymd/iprobe-sonymd/Diffuser-Super-Resolution/extracted_features_upscaled"
    print("Found the following Google ViT models:")
    print("\n".join(google_vit_models))
    
    
    # device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:4", help="Device to run the script on")
    parser.add_argument("--model_idx", type=int, help="Output directory to save the extracted features")
    args = parser.parse_args()
    
    args.device = torch.device(args.device)
    
    vit_model = google_vit_models[args.model_idx]
    checkpoint_path = os.path.join("experiments", vit_model, "best.ckpt")
    
    model_arch = vit_model.replace("google_", "google/")
    output_dir = os.path.join(output_dir, model_arch)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Extracting features for model: {model_arch}")
    main(model_arch=model_arch, model_path=checkpoint_path, output_dir = output_dir, device=args.device)
    
    # Clear the GPU memory
    torch.cuda.empty_cache()
    print("GPU memory cleared\n\n")
    
    

        
        
        
    
    
