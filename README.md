# IsoPath: Comparing Computational Pathology Foundation Models using Representational Similarity Analysis

This repository contains code for comparing various pre-trained foundational models on computational pathology tasks using Representational Similarity Analysis (RSA). The analysis focuses on Whole Slide Images (WSIs) from The Cancer Genome Atlas (TCGA) for four cancer subtypes: Breast Invasive Carcinoma (BRCA), Colon Adenocarcinoma (COAD), Lung Adenocarcinoma (LUAD), and Lung Squamous Cell Carcinoma (LUSC).

## Overview

The primary goal of this project is to understand how different state-of-the-art vision models represent histopathology image patches from different cancer types. We achieve this by:

1.  Preprocessing TCGA WSIs to extract relevant image patches.
2.  Generating feature embeddings for these patches using multiple pre-trained models.
3.  Calculating Representational Dissimilarity Matrices (RDMs) for each model based on the embeddings.
4.  Comparing these RDMs using second-order RSA (Spearman correlation, Cosine similarity) to quantify the similarity between model representations.
5.  Performing additional analyses like hierarchical clustering, specificity analysis (slide-level and disease-level using Cliff's Delta), and SVD spectral analysis to further characterize the embeddings.

## Models Compared

The following pre-trained models are included in the comparison:

*   **CONCH:** `hf_hub:MahmoodLab/conch` (ViT-B-16 based)
*   **PLIP:** `vinid/plip` (CLIP based)
*   **Prov-GigaPath:** `hf_hub:prov-gigapath/prov-gigapath`
*   **QuiltNet:** `wisdomik/QuiltNet-B-32` (CLIP based)
*   **ResNet50:** `timm` implementation, ImageNet pre-trained.
*   **UNI:** `hf-hub:MahmoodLab/UNI2-h`
*   **Virchow:** `hf-hub:paige-ai/Virchow2`

## Repository Structure

```
└── vmath20-isopath/
    ├── README.md               # This file
    ├── FinalRDMGenerations.ipynb # Jupyter Notebook for all RSA, clustering, specificity, and spectral analyses
    ├── LICENSE                 # MIT License file
    ├── preprocessing.py        # Script for WSI loading, patch extraction, and saving
    └── generate_embeddings/    # Directory containing scripts/notebooks for generating embeddings
        ├── ConchEmbeddings.ipynb # Generates embeddings using CONCH
        ├── plip.py             # Generates embeddings using PLIP
        ├── prov.py             # Generates embeddings using Prov-GigaPath
        ├── quiltnet.py         # Generates embeddings using QuiltNet
        ├── resnet.py           # Generates embeddings using ResNet50
        ├── uni.py              # Generates embeddings using UNI
        └── virchow.py          # Generates embeddings using Virchow
```

## Workflow

1.  **Data Acquisition:** Assumes access to TCGA WSI data (`.svs` files) and corresponding metadata (e.g., `gdc_sample_sheet.tsv`). Paths in the scripts point to a specific cluster setup (`/tcga/`, `/lotterlab/`).
2.  **Preprocessing (`preprocessing.py`):**
    *   Reads metadata to identify slide paths for BRCA, COAD, LUAD, LUSC.
    *   Samples 250 slides per cancer type.
    *   For each slide:
        *   Loads the WSI using `openslide`.
        *   Performs basic tissue segmentation on a thumbnail using Otsu's thresholding.
        *   Extracts 224x224 pixel patches from the tissue regions.
        *   Randomly samples 250 patches per slide.
        *   Saves the selected patches as `.npy` files (one file per slide) to a designated directory (e.g., `/lotterlab/users/vmishra/preprocessed_patches_LUAD`).
3.  **Embedding Generation (`generate_embeddings/`):**
    *   Each script/notebook loads a specific pre-trained model (e.g., CONCH, PLIP).
    *   Loads the preprocessed `.npy` patch files for all 4 cancer types.
    *   Applies the model-specific image transformations.
    *   Processes patches in batches through the model on a GPU to generate embeddings.
    *   Saves the embeddings as `.npy` files (one file per cancer type per model, e.g., `brca_embeddings_conch.npy`) to a designated directory (e.g., `/lotterlab/users/vmishra/`).
4.  **Analysis (`FinalRDMGenerations.ipynb`):**
    *   **Batching:** Loads the full embeddings and splits them into 5 batches (50 slides x 50 patches each) for robustness analysis, saving batched embeddings (e.g., `/lotterlab/users/vmishra/batched_embeddings`).
    *   **RDM Calculation:** Calculates RDMs (Euclidean distance) for each model within each batch using `rsatoolbox` and saves them (e.g., `/lotterlab/users/vmishra/rdms`). Also calculates overall RDMs for visualization.
    *   **RDM Visualization:** Generates and saves heatmap visualizations of the RDMs for each model.
    *   **Second-Order RSA:** Compares the RDMs across models using Spearman correlation and Cosine similarity, averaging results across batches. Generates and saves similarity heatmaps.
    *   **Hierarchical Clustering:** Clusters models based on RDM similarity (using Ward linkage) and plots dendrograms.
    *   **Specificity Analysis:** Calculates Cliff's Delta to compare intra-vs-inter-slide distances ("Slide Specificity") and intra-vs-inter-disease distances ("Disease Specificity"). Saves results to CSV.
    *   **Spectral Analysis:** Performs SVD on the full embeddings for each model and plots the cumulative explained variance to compare effective dimensionality.

## Usage

**Note:** The paths in the scripts (`/tcga/`, `/lotterlab/users/vmishra/`) are hardcoded and specific to a particular compute environment. You will need to adapt these paths to your own data storage and output directories.

1.  **Setup:**
    *   Clone the repository.
    *   Install the required Python packages.
    *   Ensure you have access to the required TCGA WSI data.
    *   Configure your Hugging Face Hub token if needed.
2.  **Preprocessing:**
    *   Modify paths in `preprocessing.py` for metadata, input WSI data, and the output directory for preprocessed patches.
    *   Run the script: `python preprocessing.py`
3.  **Embedding Generation:**
    *   For each script/notebook in `generate_embeddings/`:
        *   Modify paths to load the preprocessed patches generated in step 2.
        *   Modify the output path for saving the embeddings.
        *   Ensure the correct GPU device is selected (e.g., `cuda:1`).
        *   Provide your Hugging Face token where `login()` is called.
        *   Run the script (e.g., `python generate_embeddings/resnet.py`) or execute the notebook cells.
4.  **Analysis:**
    *   Modify paths in `FinalRDMGenerations.ipynb` to point to the saved embeddings (full and batched), RDM storage, and desired output locations for figures and CSV files.
    *   Execute the cells in the notebook sequentially.

## Results

The primary results are:

*   `.npy` files containing patch embeddings for each model and cancer type.
*   `.npy` files containing batched embeddings.
*   `.npy` files containing RDM matrices for each model and batch.
*   `.png` and `.pdf` files for RDM visualizations, similarity heatmaps, dendrograms, and spectral analysis plots.
*   `.csv` files containing the results of the Slide and Disease Specificity analyses.

These are generated during the execution of the scripts and notebooks, typically saved either in the local directory or in the specified output paths (e.g., `/lotterlab/users/vmishra/rdms`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

*   Vaibhav Mishra
*   William Lotter
