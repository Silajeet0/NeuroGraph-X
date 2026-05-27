'''
    Script to parcellate the brain according to the AAL Atlas
    @author: Silajeet

    FIXES vs original:
    ─────────────────────────────────────────────────────────────────
    [CRITICAL] Region mask is now applied before saving the tensor.
    Original code extracted the bounding box of each region but kept
    ALL voxels inside that box — including voxels belonging to
    adjacent brain regions that happen to fall inside the same
    bounding box. This contaminated every node feature vector:
    the hippocampus tensor contained amygdala voxels, the insula
    contained adjacent cortical voxels, etc.

    Fix: after computing the bounding box, the atlas mask for that
    specific region is cropped to the same box and multiplied into
    the patient data, zeroing out all non-region voxels.

    [FIX 2] Minimum-size guard: regions smaller than MIN_SIZE in any
    dimension are padded with zeros to MIN_SIZE. This prevents r3d_18
    from receiving tensors that are too small for its pooling layers
    (which require ≥ 8 voxels per spatial dimension after striding).

    [FIX 3] Per-scan intensity normalisation (z-score inside the
    brain mask) before extracting regions. This removes scanner-
    dependent intensity shifts that otherwise appear as a strong
    confound in the per-region feature vectors.
'''

from nilearn import image
import nibabel as nib
import numpy as np
import torch
import xml.etree.ElementTree as ET
import os
from pathlib import Path
import re

# ─── Paths ───────────────────────────────────────────────────────────────────
ATLAS_PATH      = "./data/atlas/aal_3v2/AAL3v1.nii.gz"
INPUT_DIR       = Path("./data/PostRegistration/MCI/")
XML_PATH        = "./data/atlas/aal_3v2/AAL3v1.xml"
OUTPUT_BASE_DIR = Path("./data/CNN_Tensors/")

# Minimum voxel size in every dimension after cropping.
# r3d_18 has 4 pooling layers of stride 2 → minimum usable input is 2^3 = 8.
# Using 16 gives a comfortable margin.
MIN_SIZE = 16

# ─── Load atlas ───────────────────────────────────────────────────────────────
aal  = nib.load(ATLAS_PATH)
tree = ET.parse(XML_PATH)
root = tree.getroot()

labels = {}
for label in root.findall(".//label"):
    index = int(label.find("index").text)
    name  = label.find("name").text
    labels[index] = name

print("Starting batch processing...")

for patient_file in INPUT_DIR.rglob("*.nii.gz"):
    match = re.search(r'\d{3}_S_\d{4}_I\d+', patient_file.name)
    if not match:
        continue
    patient_id = match.group(0)

    patient_out_folder = OUTPUT_BASE_DIR / patient_id
    os.makedirs(patient_out_folder, exist_ok=True)

    print(f"\nProcessing Patient: {patient_id}")
    try:
        img = nib.load(patient_file)

        # ── Resample atlas to patient space ──────────────────────────────────
        resampled_atlas = image.resample_to_img(
            aal, img, interpolation='nearest')
        aal_data      = resampled_atlas.get_fdata()
        patient_data  = img.get_fdata().astype(np.float32)

        # ── [FIX 3] Z-score normalisation inside the brain mask ──────────────
        # Brain mask = any voxel labelled by the atlas (region_id > 0)
        brain_mask = aal_data > 0
        if brain_mask.sum() > 0:
            mu  = patient_data[brain_mask].mean()
            sig = patient_data[brain_mask].std()
            if sig > 1e-6:
                patient_data = (patient_data - mu) / sig
            # Clip to ±5 σ to remove scanner artefact spikes
            patient_data = np.clip(patient_data, -5.0, 5.0)

        # ─────────────────────────────────────────────────────────────────────
        for region_id, region_name in labels.items():

            # Boolean mask selecting only the voxels of this region
            region_mask_3d = (aal_data == region_id)

            if not np.any(region_mask_3d):
                continue

            x, y, z = np.where(region_mask_3d)
            x0, x1 = x.min(), x.max() + 1
            y0, y1 = y.min(), y.max() + 1
            z0, z1 = z.min(), z.max() + 1

            # ── [CRITICAL FIX] Crop patient data AND the region mask ──────────
            # Then multiply: zeros out all non-region voxels inside the box.
            cropped_data = patient_data[x0:x1, y0:y1, z0:z1]
            cropped_mask = region_mask_3d[x0:x1, y0:y1, z0:z1].astype(np.float32)
            cropped_region = cropped_data * cropped_mask   # ← THE FIX

            # ── [FIX 2] Pad to MIN_SIZE in each spatial dimension ─────────────
            d0, d1, d2 = cropped_region.shape
            pad_d0 = max(0, MIN_SIZE - d0)
            pad_d1 = max(0, MIN_SIZE - d1)
            pad_d2 = max(0, MIN_SIZE - d2)
            if pad_d0 > 0 or pad_d1 > 0 or pad_d2 > 0:
                cropped_region = np.pad(
                    cropped_region,
                    ((0, pad_d0), (0, pad_d1), (0, pad_d2)),
                    mode='constant', constant_values=0.0,
                )

            tnsr      = torch.tensor(cropped_region, dtype=torch.float32)
            save_path = patient_out_folder / f"{region_name}.pt"
            torch.save(tnsr, save_path)

        print(f"Finished processing patient: {patient_id}")

    except Exception as e:
        print(f"CORRUPTED DATA: Skipping {patient_id}. Error: {e}")
        continue

print("Process completed successfully")
