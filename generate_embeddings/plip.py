from PIL import Image
import os
import numpy as np
import torch
import pandas as pd
from math import ceil
from tqdm import tqdm
from huggingface_hub import login

metadata_path = "/tcga/open-access/gdc_data_portal/biospecimen/tcga_Biospecimen_SAMPLE_METADATA/2023-09-01/gdc_sample_sheet.2023-09-05.tsv"
metadata_df = pd.read_csv(metadata_path, sep='\t')
slides_df = metadata_df[metadata_df['Data Type'] == 'Slide Image']
slides_df = slides_df.sort_values(by='Project ID').reset_index(drop=True)
base_dir = '/tcga/open-access/gdc_data_portal/biospecimen/tcga_Biospecimen_FILES/'
slides_df['Full Path'] = slides_df.apply(lambda row: os.path.join(base_dir, row['File ID'], row['File Name']), axis=1)

def embed(patches, model, processor, device, batch_size=64, verbose=True):
    num_batches = ceil(len(patches) / batch_size)
    opt_embs = []

    for batch_idx in tqdm(range(num_batches), disable=not verbose):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(patches))
        batch_np = patches[start:end]

        # Convert numpy arrays to PIL Images for processing
        batch_pil = [Image.fromarray(patch.astype('uint8')) for patch in batch_np]

        batch_transformed = torch.cat([processor(images=img, return_tensors="pt")["pixel_values"] for img in batch_pil])
        
        # Move batch to device
        batch = batch_transformed.to(device)

        # Call model
        model = model.to(device)

        with torch.no_grad():
            batch_emb = model.get_image_features(batch)

        # Copy to host and append
        opt_embs.append(batch_emb.cpu())

    # Stack to contiguous array
    opt_embs = torch.cat(opt_embs, dim=0)

    return opt_embs.numpy()

num_slides = 250
num_patches_per_slide = 250
patch_size = 224

brca_embeddings = []
luad_embeddings = []
lusc_embeddings = []
coad_embeddings = []

# Replace line below with hugging face token
login(token = "YOUR_HF_TOKEN")
model = CLIPModel.from_pretrained("vinid/plip")
preprocess = CLIPProcessor.from_pretrained("vinid/plip")
model.eval()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Load preprocessed patches
preprocessed_patches_dir_brca = "preprocessed_patches_BRCA"
preprocessed_patches_dir_luad = "/lotterlab/users/vmishra/preprocessed_patches_LUAD"
preprocessed_patches_dir_lusc = "/lotterlab/users/vmishra/preprocessed_patches_LUSC"
preprocessed_patches_dir_coad = "/lotterlab/users/vmishra/preprocessed_patches_COAD"

def load_patches_brca(category_label):
    patches_list = []
    filenames = [f for f in os.listdir(preprocessed_patches_dir_brca) if category_label in f]
    for filename in filenames:
        patches = np.load(os.path.join(preprocessed_patches_dir_brca, filename))
        patches_list.append(patches)
    return np.concatenate(patches_list, axis=0) if patches_list else np.array([])

def load_patches_luad(category_label):
    patches_list = []
    filenames = [f for f in os.listdir(preprocessed_patches_dir_luad) if category_label in f]
    for filename in filenames:
        patches = np.load(os.path.join(preprocessed_patches_dir_luad, filename))
        patches_list.append(patches)
    return np.concatenate(patches_list, axis=0) if patches_list else np.array([])

def load_patches_lusc(category_label):
    patches_list = []
    filenames = [f for f in os.listdir(preprocessed_patches_dir_lusc) if category_label in f]
    for filename in filenames:
        patches = np.load(os.path.join(preprocessed_patches_dir_lusc, filename))
        patches_list.append(patches)
    return np.concatenate(patches_list, axis=0) if patches_list else np.array([])

def load_patches_coad(category_label):
    patches_list = []
    filenames = [f for f in os.listdir(preprocessed_patches_dir_coad) if category_label in f]
    for filename in filenames:
        patches = np.load(os.path.join(preprocessed_patches_dir_coad, filename))
        patches_list.append(patches)
    return np.concatenate(patches_list, axis=0) if patches_list else np.array([])

brca_patches = load_patches_brca("BRCA")
luad_patches = load_patches_luad("LUAD")
lusc_patches = load_patches_lusc("LUSC")
coad_patches = load_patches_coad("COAD")

def embed_patches(patches, model, transform, device):
    if len(patches) == 0:
        return np.array([])
    model = model.to(device)
    return embed(patches, model, transform, device)

# Embed patches
brca_embeddings = embed_patches(brca_patches, model, preprocess, device)
luad_embeddings = embed_patches(luad_patches, model, preprocess, device)
lusc_embeddings = embed_patches(lusc_patches, model, preprocess, device)
coad_embeddings = embed_patches(coad_patches, model, preprocess, device)

num_brca = len(brca_embeddings)
num_luad = len(luad_embeddings)
num_lusc = len(lusc_embeddings)
num_coad = len(coad_embeddings)

# Generate labels
brca_labels = [f"BRCA_{i+1}" for i in range(num_brca)]
luad_labels = [f"LUAD_{i+1}" for i in range(num_luad)]
lusc_labels = [f"LUSC_{i+1}" for i in range(num_lusc)]
coad_labels = [f"COAD_{i+1}" for i in range(num_coad)]

# Save embeddings for all subtypes
np.save("brca_embeddings_plip.npy", brca_embeddings)
np.save("luad_embeddings_plip.npy", luad_embeddings)
np.save("lusc_embeddings_plip.npy", lusc_embeddings)
np.save("coad_embeddings_plip.npy", coad_embeddings)
