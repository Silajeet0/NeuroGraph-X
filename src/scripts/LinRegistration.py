import os
from pathlib import Path
import ants
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
input_dir = Path("data/Preprocessed/MCI")
output_dir = Path("data/PostRegistration/MCI")
template_path = "MNI152_T1_1mm.nii.gz"

output_dir.mkdir(parents=True, exist_ok=True)

# Set CPU threads (IMPORTANT for speed)
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "8"

# -----------------------------
# Load template once
# -----------------------------
template = ants.image_read(template_path)

# -----------------------------
# Collect files
# -----------------------------
nii_files = list(input_dir.glob("*.nii.gz"))
print(f"Total files: {len(nii_files)}")

# -----------------------------
# Processing loop
# -----------------------------
for file in tqdm(nii_files):

    output_path = output_dir / file.name

    # Skip already processed
    if output_path.exists():
        continue

    try:
        # -----------------------------
        # Load image
        # -----------------------------
        img = ants.image_read(str(file))

        # -----------------------------
        # 1. Bias Field Correction
        # -----------------------------
        img_n4 = ants.n4_bias_field_correction(img)

        # -----------------------------
        # 2 + 3. Registration (Affine + Nonlinear)
        # -----------------------------
        reg = ants.registration(
            fixed=template,
            moving=img_n4,
            type_of_transform="SyN"   # includes affine + nonlinear
        )

        # -----------------------------
        # Save final warped image
        # -----------------------------
        ants.image_write(reg["warpedmovout"], str(output_path))

    except Exception as e:
        print(f"Failed: {file.name} -> {e}")

print("✅ All files processed.")