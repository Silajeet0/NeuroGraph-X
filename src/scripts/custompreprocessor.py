import os
import subprocess
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# Folder paths
# -----------------------------
input_dir = Path("data/Original/MCI")
output_dir = Path("data/Preprocessed/MCI")

# output_dir.mkdir(parents=True, exist_ok=True)

files1 = set(os.listdir(input_dir))
files2 = set(os.listdir(output_dir))
nii_files = files1 - files2

# -----------------------------
# Collect files
# -----------------------------
# nii_files=[]
# nii_files = list(input_dir.glob("*.nii.gz"))


print(f"Total files found: {len(nii_files)}")

# -----------------------------
# Processing loop
# -----------------------------
for file in tqdm(nii_files):

    input_path = str(input_dir / file)
    output_path = str(output_dir / file)

    if os.path.exists(output_path):
        print(f"Skipping (already exists): {file}")
        continue

    cmd = [
        "hd-bet",
        "-i", input_path,
        "-o", output_path,
        "-device", "cuda:0"
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed processing: {file}")

print("Processing complete.")