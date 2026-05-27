import pandas as pd
import os
import re
from pathlib import Path
import numpy as np

# 1. Paths
csv_path = "DXSUM_10Mar2026.csv" 
image_dir = Path("data/Preprocessed/") 

print("Loading clinical data...")
df = pd.read_csv(csv_path)

# 2. PATIENT ID EXTRACTION
my_patients = []
for file_path in image_dir.rglob("*.nii*"):
    # This regex specifically looks for the "XXX_S_XXXX" pattern
    match = re.search(r'\d{3}_S_\d{4}', file_path.name)
    if match:
        ptid = match.group(0)
        if ptid not in my_patients:
            my_patients.append(ptid)

print(f"Found {len(my_patients)} physical patients on disk.")

if len(my_patients) > 0:
    print(f"Sample ID from the files: {my_patients[0]}")
    print(f"Sample ID from the CSV:    {df['PTID'].iloc[0]}")

# 3. Filter the massive ADNI CSV to ONLY patients of interest
df = df[df['PTID'].isin(my_patients)].copy()

print(f"Patients successfully matched in CSV: {df['PTID'].nunique()}")

# Sort by patient-id and visit date so we can track them chronologically
df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
df = df.sort_values(by=['PTID', 'EXAMDATE'])

# 4. The Logic: Stable MCI vs Progressive MCI
results = []

print(f"Unique values inside DIAGNOSIS column: {df['DIAGNOSIS'].dropna().unique()}")

for ptid, patient_data in df.groupby('PTID'):
    dx_codes = patient_data['DIAGNOSIS'].dropna().values
    
    if len(dx_codes) == 0:
        continue
        
    baseline_dx = dx_codes[0]
    hit_ad = False
    valid_mci_patient = False
    
    # ADNI is using numbers (1=CN, 2=MCI, 3=AD)
    if isinstance(baseline_dx, (int, float, np.integer, np.floating)):
        if baseline_dx == 2:  # Started as MCI
            valid_mci_patient = True
            hit_ad = any(code == 3 for code in dx_codes) # Progressed to AD (3)

    # Apply the labels for NeuroGraph-X
    if valid_mci_patient:
        if hit_ad:
            label = 1  # Progressive MCI (Target Class!)
            status = "pMCI"
        else:
            label = 0  # Stable MCI
            status = "sMCI"
            
        results.append({"PTID": ptid, "Label": label, "Status": status})

# 5. Save and Print
if len(results) == 0:
    print("\nERROR: Still found 0 MCI patients. We need to check the DXCHANGE codes.")
else:
    final_labels_df = pd.DataFrame(results)
    final_labels_df.to_csv("GNN_Target_Labels.csv", index=False)
    print(f"\nLabelling Complete! Here is your class distribution:")
    print(final_labels_df['Status'].value_counts())