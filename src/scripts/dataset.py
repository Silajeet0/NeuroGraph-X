'''
    PyG InMemoryDataset for NeuroGraph-X.

    FIXES vs original:
    ─────────────────────────────────────────────────────────────────
    [FIX 1] Edge validation: if get_Morphological_Features returns 0
    edges (all cosine similarities below threshold — can happen for
    patients with many missing regions) a self-loop fallback is used
    so the GATv2 message passing does not receive an empty edge tensor.

    [FIX 2] Optional x_global support: if ./data/Global_Features/
    exists and contains a .pt file matching the patient ID, the global
    AD-pretrained embedding is stored as `x_global` in the Data object.
    This is consumed by BrainGATv2 and concatenated with h_fused before
    the final classifier (see train_v5.ipynb for the updated forward()).
    If no global feature file is found the field is omitted — the
    dataset and training script are backward-compatible.

    [FIX 3] Robust PTID extraction: handles both XXX_S_XXXX_IYYY and
    XXX_S_XXXX folder naming conventions.
'''

import os
import re
import torch
import pandas as pd
from pathlib import Path
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

from utils import get_Patient_Matrix, get_Morphological_Features

GLOBAL_FEAT_DIR = Path("./data/Global_Features/")


class NeuroGraphDataset(InMemoryDataset):
    def __init__(self, root, csv_path, spatial_matrix_path,
                 transform=None, pre_transform=None):
        self.csv_path            = csv_path
        self.spatial_matrix_path = spatial_matrix_path
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['neurograph_processed.pt']

    def process(self):
        data_list = []

        print("Loading ADNI labels...")
        df = pd.read_csv(self.csv_path)
        label_dict = {str(row['PTID']): int(row['Label'])
                      for _, row in df.iterrows()}

        print("Loading global Spatial Matrix...")
        spatial      = torch.load(self.spatial_matrix_path, weights_only=False)
        ei_spatial   = spatial['edge_index']
        ea_spatial   = spatial['edge_attr']

        raw_features_dir = os.path.join(self.root, 'raw', 'GNN_Features')
        patient_folders  = [
            f for f in os.listdir(raw_features_dir)
            if os.path.isdir(os.path.join(raw_features_dir, f))
        ]

        use_global = GLOBAL_FEAT_DIR.exists() and any(GLOBAL_FEAT_DIR.glob("*.pt"))
        if use_global:
            print(f"Global feature directory found — will attach x_global.")

        no_label   = 0
        no_morph   = 0
        built      = 0

        print("Building PyG Data objects...")
        for subject_folder in tqdm(patient_folders):
            patient_path = os.path.join(raw_features_dir, subject_folder)

            # ── [FIX 3] Robust PTID extraction ──────────────────────────────
            match = re.search(r'(\d{3}_S_\d{4})', subject_folder)
            if not match:
                continue
            ptid = match.group(1)

            if ptid not in label_dict:
                no_label += 1
                continue

            y_label  = torch.tensor([label_dict[ptid]], dtype=torch.float32)
            x_matrix = get_Patient_Matrix(patient_path)

            # Morphological edges
            ei_morph, ea_morph = get_Morphological_Features(
                patient_matrix=x_matrix, threshold=0.75)

            # ── [FIX 1] Edge validation fallback ────────────────────────────
            if ei_morph.size(1) == 0:
                no_morph += 1
                # Self-loops as fallback so GAT doesn't receive empty graph
                idx      = torch.arange(x_matrix.size(0))
                ei_morph = torch.stack([idx, idx], dim=0)
                ea_morph = torch.ones(x_matrix.size(0), dtype=torch.float32)

            patient_data = Data(
                x                  = x_matrix,
                edge_index_spatial = ei_spatial,
                edge_attr_spatial  = ea_spatial,
                edge_index_morph   = ei_morph,
                edge_attr_morph    = ea_morph,
                y                  = y_label,
            )

            # ── [FIX 2] Attach global embedding if available ─────────────────
            if use_global:
                # File may be named by PTID_ImageID or just PTID
                matches = list(GLOBAL_FEAT_DIR.glob(f"{ptid}*.pt"))
                if not matches:
                    # Try folder name directly
                    direct = GLOBAL_FEAT_DIR / f"{subject_folder}.pt"
                    if direct.exists():
                        matches = [direct]
                if matches:
                    try:
                        gf = torch.load(matches[0], weights_only=False)
                        gf = gf.squeeze().float()
                        if torch.isfinite(gf).all():
                            patient_data.x_global = gf
                    except Exception:
                        pass

            data_list.append(patient_data)
            built += 1

        print(f"\nDataset built: {built} patients")
        if no_label:
            print(f"  Skipped (no label in CSV): {no_label}")
        if no_morph:
            print(f"  Used self-loop fallback (empty morph graph): {no_morph}")
        has_global = sum(1 for d in data_list if hasattr(d, 'x_global'))
        if use_global:
            print(f"  Patients with x_global: {has_global}/{built}")

        # Zero-fill any missing x_global so collate doesn't crash
        if use_global:
            global_dim = next(
                (d.x_global.shape[0] for d in data_list if hasattr(d, 'x_global')),
                256
            )
            for d in data_list:
                if not hasattr(d, 'x_global'):
                    d.x_global = torch.zeros(global_dim, dtype=torch.float32)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    print("Initializing NeuroGraphDataset...")
    dataset = NeuroGraphDataset(
        root                 = './data',
        csv_path             = './data/GNN_Target_Labels.csv',
        spatial_matrix_path  = './data/A_spatial_master.pt',
    )
    print(f"\nTotal patients: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample graph: {sample}")
    if hasattr(sample, 'x_global'):
        print(f"x_global shape: {sample.x_global.shape}")
