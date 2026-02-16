import os
import pandas as pd
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "ADNI_Raw_Data", "ADNI")
DATA_LABEL_CSV = os.path.join(PROJECT_ROOT, "NeuroGraph_Raw_2_15_2026.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data")

def organize_and_convert():
    df = pd.read_csv(DATA_LABEL_CSV)
    df.columns = df.columns.str.strip()
    print(df.head())

    for group in ["AD", "CN", "MCI"]:
        os.makedirs(os.path.join(OUTPUT_PATH, group), exist_ok=True)

    print(f"Scanning {RAW_DATA_PATH} to group items")

    count = 0
    for root, dirs, files in os.walk(RAW_DATA_PATH):
        if any(f.endswith(".dcm") for f in files):
            
            # Get Image ID from path
            # Path usually ends like: .../2022-09-09_12_12_12/I1619403
            parts = root.split(os.sep)
            image_id = parts[-1] 
            
            # --- CRITICAL FIX: MATCHING THE EXACT COLUMN NAME ---
            # Your screenshot shows "Image Data ID", so we look for that.
            row = df[df['Image Data ID'] == image_id]
            
            if row.empty:
                continue
            
            # Get Class and Subject
            group = row.iloc[0]['Group']
            subject_id = row.iloc[0]['Subject'] # Screenshot says 'Subject', not 'Subject ID'
            
            # Merge Groups
            if group in ["EMCI", "LMCI"]: group = "MCI"
            if group not in ["AD", "CN", "MCI"]: continue

            # Destination
            dest_folder = os.path.join(OUTPUT_PATH, group)
            dest_filename = f"{subject_id}_{image_id}"
            
            # Convert
            cmd = [
                "dcm2niix", "-z", "y", "-b", "n", 
                "-f", dest_filename, 
                "-o", dest_folder, 
                root
            ]
            
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            count += 1
            if count % 10 == 0:
                print(f"Converted {count} scans...")

    print(f"\n--- DONE! ---")
    print(f"Total processed: {count}")

if __name__ == "__main__":
    organize_and_convert()