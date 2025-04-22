# IsoPath
This repository contains code for preprocessing Whole Slide Images (WSIs) from The Cancer Genome Atlas (TCGA) and generating feature embeddings using various deep learning models. The primary focus is on processing slides from specific cancer cohorts (BRCA, COAD, LUAD, LUSC) and extracting patch-level embeddings.

## Overview

The workflow involves two main steps:

1.  **Preprocessing:** Reading TCGA WSI files, performing tissue segmentation, extracting relevant image patches, and saving them.
2.  **Embedding Generation:** Loading the preprocessed patches and using different pre-trained models (e.g., ResNet, CONCH, UNI, Virchow, etc.) to generate feature embeddings for each patch.

The generated embeddings can be used for various downstream tasks in computational pathology, such as subtype classification, survival analysis, or exploring morphological similarities across different models and datasets.

## Features

*   Preprocessing pipeline for TCGA WSIs using `openslide` and `scikit-image`.
*   Tissue segmentation using Otsu's thresholding.
*   Patch extraction and sampling.
*   Scripts to generate embeddings using multiple state-of-the-art pathology foundation models:
    *   CONCH
    *   PLIP
    *   Prov-GigaPath
    *   QuiltNet
    *   ResNet50 (ImageNet pre-trained baseline)
    *   UNI
    *   Virchow

## Directory Structure

```
vmath20-isopath/
├── README.md               # This README file
├── LICENSE                 # Project license information (MIT)
├── preprocessing.py        # Script for WSI preprocessing and patch extraction
└── generate_embeddings/    # Scripts for generating embeddings using different models
    ├── ConchEmbeddings.ipynb # Original notebook for CONCH (converted to .py for execution)
    ├── ConchEmbeddings.py    # Python script version of the CONCH notebook
    ├── plip.py               # Script to generate embeddings using PLIP
    ├── prov.py               # Script to generate embeddings using Prov-GigaPath
    ├── quiltnet.py           # Script to generate embeddings using QuiltNet
    ├── resnet.py             # Script to generate embeddings using ResNet50
    ├── uni.py                # Script to generate embeddings using UNI
    └── virchow.py            # Script to generate embeddings using Virchow
```

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vmath20/isopath.git
    cd vmath20-isopath
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment (e.g., `conda` or `venv`).
    ```bash
    # Example using conda
    conda create -n isopath python=3.10
    conda activate isopath
    ```

    Install necessary libraries. You might need system libraries for `openslide-python`. On Debian/Ubuntu: `sudo apt-get install openslide-tools libopenslide-dev`

    Install Python packages:
    ```bash
    pip install openslide-python Pillow numpy torch torchvision tifffile pandas scikit-image tqdm matplotlib seaborn scikit-learn timm huggingface_hub einops transformers
    ```

    **Special Installation for CONCH:**
    CONCH needs to be installed directly from its repository:
    ```bash
    pip install git+https://github.com/Mahmoodlab/CONCH.git
    ```
    *Note: You might need to run `rm generate_embeddings/conch.py` if you encounter import issues related to a potentially existing `conch.py` file after installation.*

3.  **Hugging Face Login:**
    Several models are downloaded from the Hugging Face Hub. You might need to log in using your Hugging Face token, especially for models requiring authentication (like CONCH initially, though the scripts use specific hub paths). The scripts contain placeholders like `login(token="YOUR_HF_TOKEN")`. Replace `"YOUR_HF_TOKEN"` with your actual token or use the Hugging Face CLI:
    ```bash
    huggingface-cli login
    ```

## Usage

### 1. Data Preparation

*   **TCGA Data:** The scripts assume access to TCGA WSI data and metadata. The paths in the scripts (e.g., `/tcga/open-access/...`, `/lotterlab/users/vmishra/...`) are specific to the original development environment. **You MUST modify these paths** in `preprocessing.py` and all scripts within `generate_embeddings/` to point to your local TCGA data locations and desired output directories.
*   **Metadata:** Ensure the `metadata_path` in `preprocessing.py` points to the correct TCGA sample sheet TSV file.
*   **WSI Files:** Ensure the `base_dir` in `preprocessing.py` points to the root directory containing the TCGA WSI files organized by File ID.

### 2. Preprocessing WSIs

Run the `preprocessing.py` script to process the WSIs, extract patches, and save them as `.npy` files.

```bash
python preprocessing.py
```

This script will:
*   Read the metadata file.
*   Filter for 'Slide Image' data type.
*   Sample `num_slides` (default: 250) slides for each specified cancer type (COAD, BRCA, LUSC, LUAD).
*   For each sampled slide:
    *   Read the WSI using `openslide`.
    *   Segment tissue using `skimage`.
    *   Extract non-overlapping patches of `patch_size` (default: 224x224).
    *   Randomly sample `num_patches_per_slide` (default: 250) patches from the tissue regions.
    *   Save the selected patches as a `.npy` file in the specified `preprocessed_patches_dir` for that cancer type.

### 3. Generating Embeddings

Navigate to the `generate_embeddings/` directory or run the scripts from the root directory. Each script corresponds to a specific model.

**Before running:** Ensure the paths for loading preprocessed patches (`preprocessed_patches_dir_*`) and saving embeddings (`np.save(...)` paths) within each script are correctly set for your environment. Also, ensure the correct GPU device is selected (e.g., `cuda:0`, `cuda:1`).

**Example (running ResNet embedding generation):**
```bash
python generate_embeddings/resnet.py
```

**Example (running CONCH embedding generation):**
```bash
python generate_embeddings/ConchEmbeddings.py
```

Repeat this process for each model script (`plip.py`, `prov.py`, `quiltnet.py`, `uni.py`, `virchow.py`) you want to generate embeddings for. Each script will load the corresponding `.npy` patch files, process them through the model, and save the resulting embeddings as separate `.npy` files (e.g., `brca_embeddings_resnet.npy`, `luad_embeddings_conch.npy`, etc.).

## Models Used

*   **CONCH:** `MahmoodLab/conch` (via `conch` library)
*   **PLIP:** `vinid/plip` (via `transformers`)
*   **Prov-GigaPath:** `prov-gigapath/prov-gigapath` (via `timm`)
*   **QuiltNet:** `wisdomik/QuiltNet-B-32` (via `transformers`)
*   **ResNet50:** `resnet50` (ImageNet pretrained via `timm`)
*   **UNI:** `MahmoodLab/UNI2-h` (via `timm`)
*   **Virchow:** `paige-ai/Virchow2` (via `timm`)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Author

*   Vaibhav Mishra
*   William Lotter
