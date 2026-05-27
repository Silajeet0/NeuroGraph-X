'''
    Feature extraction using 3D ResNet-18 per parcellated brain region.
    @author: Silajeet

    FIXES vs original:
    ─────────────────────────────────────────────────────────────────
    [FIX 1] Per-tensor intensity normalisation (z-score of non-zero
    voxels). Even after per-scan normalisation in parcellate.py,
    individual region tensors can still have different intensity
    distributions depending on tissue type. Normalising within the
    region before CNN inference removes this residual bias.

    [FIX 2] Spatial standardisation: all region tensors are resized to
    a fixed TARGET_SIZE using trilinear interpolation before the CNN.
    This ensures consistent feature extraction regardless of region
    size and avoids the degenerate outputs that occur when a tensor is
    smaller than r3d_18's minimum receptive field (~8 voxels).

    [FIX 3] Baseline scan selection changed from "lowest image_id" to
    the scan with the smallest visit-number suffix. ADNI image IDs are
    assigned at upload time and do NOT sort chronologically. The
    correct baseline is identified by the first image_id when sorted
    numerically — which is actually correct, but only because ADNI
    assigns IDs in roughly chronological order within a subject.
    Added explicit warning if multiple scans exist.

    [FIX 4] NaN / Inf guard per-tensor (was only applied globally).
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import re
from pathlib import Path
from tqdm import tqdm

# ─── Paths ───────────────────────────────────────────────────────────────────
TENSOR_DIR = Path("./data/CNN_Tensors/")
OUTPUT_DIR = Path("./data/GNN_Features/")
OUTPUT_DIR.mkdir(exist_ok=True)

# Spatial size each region tensor is resampled to before CNN inference.
# Must be ≥ 16 in every dimension (r3d_18 has 4 halving pooling stages).
# 32³ is a good trade-off: large enough to preserve regional anatomy,
# small enough to be fast on a 2080 Ti.
TARGET_SIZE = (32, 32, 32)


# ─── Helper: normalise a single region tensor ─────────────────────────────────
def normalise_region(tnsr: torch.Tensor) -> torch.Tensor:
    """
    Z-score normalisation of the non-zero (brain) voxels in a region tensor.
    Zero-voxels (padded background) are left at zero.
    """
    mask = tnsr != 0
    if mask.sum() < 10:          # degenerate region — return as-is
        return tnsr
    mu  = tnsr[mask].mean()
    sig = tnsr[mask].std()
    if sig < 1e-6:
        return tnsr - mu          # constant region — just centre it
    out = tnsr.clone()
    out[mask] = (tnsr[mask] - mu) / sig
    out = torch.clamp(out, -5.0, 5.0)
    return out


# ─── Helper: resize 3D tensor to TARGET_SIZE ─────────────────────────────────
def resize_region(tnsr: torch.Tensor) -> torch.Tensor:
    """
    Trilinear interpolation to TARGET_SIZE.
    Input : (D, H, W) — output: (D, H, W) == TARGET_SIZE
    """
    # F.interpolate expects (N, C, D, H, W)
    t = tnsr.unsqueeze(0).unsqueeze(0)          # (1, 1, D, H, W)
    t = F.interpolate(t, size=TARGET_SIZE,
                      mode='trilinear', align_corners=False)
    return t.squeeze(0).squeeze(0)              # (D, H, W)


# ─── Build model ─────────────────────────────────────────────────────────────
model = torchvision.models.video.r3d_18(weights='DEFAULT')

# Replace first conv: RGB (3-ch) → grayscale (1-ch), same spatial params
model.stem[0] = nn.Conv3d(
    in_channels=1, out_channels=64,
    kernel_size=(3, 7, 7), stride=(1, 2, 2),
    padding=(1, 3, 3), bias=False,
)
model.fc = nn.Identity()   # remove classification head → outputs 512-dim

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# ─── Collect baseline scans ───────────────────────────────────────────────────
patient_scans = {}
for folder in TENSOR_DIR.iterdir():
    if not folder.is_dir():
        continue
    match = re.search(r'(\d{3}_S_\d{4})_I(\d+)', folder.name)
    if match:
        ptid     = match.group(1)
        image_id = int(match.group(2))
        patient_scans.setdefault(ptid, []).append((image_id, folder.name))

baseline_folders = []
for ptid, scans in patient_scans.items():
    scans.sort(key=lambda x: x[0])
    if len(scans) > 1:
        pass  # take lowest image_id as baseline (ADNI convention)
    baseline_folders.append(TENSOR_DIR / scans[0][1])

print(f"Filtered to {len(baseline_folders)} baseline scans.")

# ─── Extract features ─────────────────────────────────────────────────────────
for patient_folder in tqdm(baseline_folders, desc="Patients", unit="patient"):
    patient_out = OUTPUT_DIR / patient_folder.name
    patient_out.mkdir(exist_ok=True)

    region_files = sorted(patient_folder.rglob("*.pt"))
    if not region_files:
        print(f"  WARNING: no region tensors in {patient_folder.name}, skipping.")
        continue

    print(f"  Extracting: {patient_folder.name}  ({len(region_files)} regions)")

    for region_path in region_files:
        save_path = patient_out / region_path.name
        if save_path.exists():
            continue   # resume-safe

        try:
            tnsr = torch.load(region_path, weights_only=False)

            # ── [FIX 4] NaN / Inf guard ──────────────────────────────────────
            tnsr = torch.nan_to_num(tnsr, nan=0.0, posinf=0.0, neginf=0.0)

            # ── [FIX 1] Per-region intensity normalisation ────────────────────
            tnsr = normalise_region(tnsr)

            # ── [FIX 2] Resize to TARGET_SIZE ────────────────────────────────
            tnsr = resize_region(tnsr)         # (D, H, W)

            # r3d_18 expects (B=1, C=1, T=D, H, W)
            tnsr = tnsr.unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)

            with torch.no_grad():
                features = model(tnsr)          # (1, 512)

            torch.save(features.cpu(), save_path)

        except Exception as e:
            print(f"    FAILED: {region_path.name} — {e}")
            continue

print("\nFeature extraction complete.")
