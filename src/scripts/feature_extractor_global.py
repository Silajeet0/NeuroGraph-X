'''
    Whole-brain feature extraction using the AD-pretrained 3D CNN
    from Turrisi et al. (2024) "Deep Learning-based Alzheimer's Disease
    detection: reproducibility and the effect of modeling choices."
    Frontiers in Computational Neuroscience.
    https://github.com/rturrisige/3D_CNN_pretrained_model

    WHY this is better than per-region r3d_18 as a global descriptor:
    ─────────────────────────────────────────────────────────────────
    • Pretrained on ADNI T1-weighted MRI — the same dataset and scanner
      protocol as your patients. Domain match is exact.
    • Supervised for AD/CN discrimination: the learned features directly
      encode neurodegeneration-relevant patterns (hippocampal atrophy,
      cortical thinning, ventricular enlargement).
    • r3d_18 was pretrained on Kinetics (sports videos) — zero domain
      overlap with brain MRI.

    OUTPUT:
    ─────────────────────────────────────────────────────────────────
    For each patient, saves:
        ./data/Global_Features/<PTID_ImageID>.pt   shape: (256,)

    This global embedding is later loaded by dataset.py and stored as
    `x_global` in the PyG Data object. The GNN training script
    concatenates it with h_fused before the final classifier.

    SETUP (run once before this script):
    ─────────────────────────────────────────────────────────────────
        git clone https://github.com/rturrisige/3D_CNN_pretrained_model \
            ./ad_pretrained_model
    The weights file AD_pretrained_weights.pt and utilities file
    AD_pretrained_utilities.py are then available locally.
'''

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────────────────────
POST_REG_DIR   = Path("./data/PostRegistration/MCI/")
OUTPUT_DIR     = Path("./data/Global_Features/")
AD_MODEL_DIR   = Path("./ad_pretrained_model")        # cloned repo
WEIGHTS_PATH   = AD_MODEL_DIR / "AD_pretrained_weights.pt"
UTILITIES_PATH = AD_MODEL_DIR / "AD_pretrained_utilities.py"

OUTPUT_DIR.mkdir(exist_ok=True)

# ─── Sanity check ─────────────────────────────────────────────────────────────
if not WEIGHTS_PATH.exists() or not UTILITIES_PATH.exists():
    raise FileNotFoundError(
        "AD-pretrained model not found. Run:\n"
        "  git clone https://github.com/rturrisige/3D_CNN_pretrained_model "
        "./ad_pretrained_model\n"
        "then re-run this script."
    )

# ─── Import the CNN class from cloned repo ────────────────────────────────────
sys.path.insert(0, str(AD_MODEL_DIR))
from ad_pretrained_model.AD_pretrained_utilities import CNN, CNN_8CL_B   # noqa: E402
    
# ─── Load pretrained weights, strip classification head ──────────────────────
ad_cnn = CNN(CNN_8CL_B())
state  = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
ad_cnn.load_state_dict(state, strict=False)

# Remove the final classification layer to expose the embedding.
# The CNN class from rturrisige typically ends with a Linear classifier.
# We replace it with Identity so forward() returns the penultimate features.
if hasattr(ad_cnn, 'fc'):
    embedding_dim = ad_cnn.fc.in_features
    ad_cnn.fc    = nn.Identity()
elif hasattr(ad_cnn, 'classifier'):
    # Some variants wrap the head in a Sequential
    last = list(ad_cnn.classifier.children())[-1]
    embedding_dim = last.in_features
    ad_cnn.classifier[-1] = nn.Identity()
else:
    # Fallback: hook the last Linear layer
    embedding_dim = None
    for name, m in reversed(list(ad_cnn.named_modules())):
        if isinstance(m, nn.Linear):
            embedding_dim = m.in_features
            # Replace with Identity via setattr on parent module
            parts  = name.split(".")
            parent = ad_cnn
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], nn.Identity())
            break

print(f"AD-pretrained CNN loaded. Embedding dim: {embedding_dim}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ad_cnn.eval().to(device)


# ─── MRI preprocessing ────────────────────────────────────────────────────────
# The rturrisige model expects a whole-brain T1 volume. Standard practice
# for ADNI models: resample to a fixed voxel grid, z-score normalise.
TARGET_SHAPE = (73, 96, 96)   # typical MNI152 2mm bounding box


def preprocess_nifti(nii_path: Path) -> torch.Tensor:
    """
    Load a post-registration NIfTI, resample to TARGET_SHAPE,
    z-score normalise, return (1, 1, D, H, W) tensor.
    """
    img  = nib.load(str(nii_path))
    data = img.get_fdata().astype(np.float32)

    # Resample via trilinear (on CPU with F.interpolate)
    t = torch.tensor(data).unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)
    t = F.interpolate(t, size=TARGET_SHAPE,
                      mode='trilinear', align_corners=False)

    # Z-score normalise non-zero (brain) voxels
    mask = t > 0
    if mask.sum() > 100:
        mu  = t[mask].mean()
        sig = t[mask].std()
        if sig > 1e-6:
            t[mask] = (t[mask] - mu) / sig
    t = torch.clamp(t, -5.0, 5.0)
    return t   # (1, 1, D, H, W)


# ─── Collect baseline NIfTI files ─────────────────────────────────────────────
patient_scans: dict = {}
for nii_file in POST_REG_DIR.rglob("*.nii.gz"):
    match = re.search(r'(\d{3}_S_\d{4})_I(\d+)', nii_file.name)
    if not match:
        continue
    ptid     = match.group(1)
    image_id = int(match.group(2))
    patient_scans.setdefault(ptid, []).append((image_id, nii_file))

baseline_files = []
for ptid, scans in patient_scans.items():
    scans.sort(key=lambda x: x[0])
    baseline_files.append(scans[0][1])   # lowest image_id = baseline

print(f"Found {len(baseline_files)} baseline scans for global feature extraction.")

# ─── Extract and save ─────────────────────────────────────────────────────────
failed = []
for nii_path in tqdm(baseline_files, desc="Global features", unit="patient"):
    match = re.search(r'\d{3}_S_\d{4}_I\d+', nii_path.name)
    if not match:
        continue
    patient_id = match.group(0)
    save_path  = OUTPUT_DIR / f"{patient_id}.pt"

    if save_path.exists():
        continue

    try:
        vol = preprocess_nifti(nii_path).to(device, dtype=torch.float32)
        with torch.no_grad():
            emb = ad_cnn(vol)              # (1, embedding_dim)
        torch.save(emb.squeeze(0).cpu(), save_path)   # (embedding_dim,)
    except Exception as e:
        print(f"  FAILED: {patient_id} — {e}")
        failed.append(patient_id)

print(f"\nGlobal feature extraction complete.")
print(f"  Saved: {len(list(OUTPUT_DIR.glob('*.pt')))} files")
if failed:
    print(f"  Failed ({len(failed)}): {failed[:5]}{'...' if len(failed) > 5 else ''}")
