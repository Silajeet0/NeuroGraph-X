'''
    Generate the master spatial adjacency matrix from AAL atlas centroids.

    FIX: MASTER_TEMPLATE is no longer hard-coded to a single patient file.
    Any valid NIfTI in the post-registration directory is used. This prevents
    crashes when the original hard-coded patient is not present.
'''

import torch
import numpy as np
import nibabel as nib
from nilearn import image
import json
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

with open("./data/region_mapping.json", "r") as f:
    REGION_TO_IDX = json.load(f)

POST_REG_DIR = Path("./data/PostRegistration/MCI/")


def _pick_template() -> Path:
    """Return any valid NIfTI from the post-registration directory."""
    candidates = sorted(POST_REG_DIR.rglob("*.nii.gz"))
    if not candidates:
        raise FileNotFoundError(
            f"No NIfTI files found in {POST_REG_DIR}. "
            "Run registration before generating the spatial matrix."
        )
    return candidates[0]


def get_Spatial_Features():
    K_NEIGHBORS = 5
    RBF_SIGMA   = 50.0
    NUM_NODES   = 166
    ATLAS_PATH  = "./data/atlas/aal_3v2/AAL3v1.nii.gz"

    nifti_to_name = {}
    with open("./data/atlas/aal_3v2/AAL3v1.nii.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                nifti_to_name[int(parts[0])] = parts[1]

    atlas    = nib.load(ATLAS_PATH)
    template = _pick_template()
    print(f"Using template: {template.name}")
    ref_img  = nib.load(str(template))

    print("Resampling AAL Atlas to template space...")
    resampled = image.resample_to_img(atlas, ref_img, interpolation='nearest')
    mask_data = resampled.get_fdata()
    affine    = resampled.affine

    print("Extracting physical centroids...")
    centroids_mm = np.zeros((NUM_NODES, 3))

    for nifti_id, region_name in nifti_to_name.items():
        if region_name not in REGION_TO_IDX:
            continue
        target_row  = REGION_TO_IDX[region_name]
        coords      = np.argwhere(mask_data == nifti_id)
        if len(coords) == 0:
            print(f"  WARNING: Region {region_name} has no voxels!")
            continue
        centroid_v  = coords.mean(axis=0)
        centroid_mm = affine.dot(np.append(centroid_v, 1.0))[:3]
        centroids_mm[target_row] = centroid_mm

    print("Computing KNN + RBF spatial adjacency...")
    dist_mtx = squareform(pdist(centroids_mm, metric='euclidean'))

    knn_mask = np.zeros_like(dist_mtx)
    for i in range(NUM_NODES):
        nn_idx = np.argsort(dist_mtx[i])[1:K_NEIGHBORS + 1]
        knn_mask[i, nn_idx] = 1
        knn_mask[nn_idx, i] = 1

    rbf_weights  = np.exp(-(dist_mtx ** 2) / (2 * RBF_SIGMA ** 2))
    A_spatial    = rbf_weights * knn_mask

    pos_mask     = A_spatial > 0
    edge_index   = torch.nonzero(torch.tensor(pos_mask), as_tuple=False).t().contiguous()
    edge_attr    = torch.tensor(A_spatial[pos_mask], dtype=torch.float32)

    out_path = "./data/A_spatial_master.pt"
    torch.save({'edge_index': edge_index, 'edge_attr': edge_attr}, out_path)
    print(f"Saved spatial matrix → {out_path}  "
          f"({edge_index.size(1)} edges)")


print("=== Generating Master Spatial Matrix ===")
get_Spatial_Features()
