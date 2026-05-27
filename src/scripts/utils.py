import torch
import numpy as np
import torch.nn.functional as F
import json
import os

with open("./data/region_mapping.json", "r") as f:
    REGION_TO_IDX = json.load(f)


def get_Patient_Matrix(patient_folder_path):
    """
    Build the (N_regions, 512) node feature matrix for one patient.

    FIX: Missing regions were previously left as zero rows. After L2
    normalisation a zero row becomes 0/0 = NaN, which propagates through
    the entire GNN via message passing. Missing regions are now filled with
    the mean of the present regions so the graph remains well-defined.
    """
    patient_matrix = torch.zeros((166, 512))
    present_mask   = torch.zeros(166, dtype=torch.bool)

    for filename in os.listdir(patient_folder_path):
        if not filename.endswith(".pt"):
            continue
        region_name = filename.replace(".pt", "")
        if region_name not in REGION_TO_IDX:
            continue
        row_idx = REGION_TO_IDX[region_name]
        try:
            fv = torch.load(
                os.path.join(patient_folder_path, filename),
                weights_only=False,
            )
            fv = fv.squeeze()
            if fv.shape != (512,):
                continue
            # Guard individual NaN / Inf values from CNN output
            if not torch.isfinite(fv).all():
                fv = torch.nan_to_num(fv, nan=0.0, posinf=0.0, neginf=0.0)
            patient_matrix[row_idx] = fv
            present_mask[row_idx]   = True
        except Exception:
            continue

    # Fill missing regions with the mean of present regions
    # to avoid NaN after L2 normalisation
    if present_mask.sum() > 0:
        mean_vec = patient_matrix[present_mask].mean(dim=0)
        patient_matrix[~present_mask] = mean_vec

    return patient_matrix


def get_Morphological_Features(patient_matrix, threshold=0.75):
    """
    Build the morphological adjacency graph from cosine similarity.

    FIX: Added adaptive threshold fallback. If the fixed threshold produces
    fewer than (N_regions * 2) edges (essentially a disconnected graph),
    the threshold is lowered in steps of 0.05 until a connected graph is
    obtained or a minimum threshold of 0.50 is reached.

    A disconnected morphological graph causes the GAT to only aggregate
    information within disconnected components, making the stream useless.
    """
    # L2 normalise — safe after mean-fill above (no zero rows)
    normalized_mtx = F.normalize(patient_matrix, p=2, dim=1)
    cosine_mtx     = torch.mm(normalized_mtx, normalized_mtx.t())

    N           = patient_matrix.size(0)
    min_edges   = N * 2          # at least avg degree 2
    current_thr = threshold

    while current_thr >= 0.50:
        mask = cosine_mtx >= current_thr
        mask.fill_diagonal_(False)
        n_edges = mask.sum().item()
        if n_edges >= min_edges:
            break
        current_thr -= 0.05

    if current_thr < threshold:
        pass  # threshold was adapted; caller need not know

    edge_index = torch.nonzero(mask, as_tuple=False).t().contiguous()
    edge_attr  = cosine_mtx[mask].clone().detach().to(torch.float32)

    return edge_index, edge_attr
