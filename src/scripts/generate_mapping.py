import os
import json

def generate_alphabetical_mapping(sample_patient_folder):
    #get all files and sort them alphabetically
    filenames = sorted(os.listdir(sample_patient_folder))
    
    region_to_idx = {}
    current_idx = 0
    
    for filename in filenames:
        if filename.endswith(".pt"):
            region_name = filename.replace(".pt", "")
            region_to_idx[region_name] = current_idx
            current_idx += 1
            
    #store it in a JSON file
    with open("region_mapping.json", "w") as out:
        json.dump(region_to_idx, out, indent=4)
        
    print(f"Alphabetical map saved with {len(region_to_idx)} regions.")

generate_alphabetical_mapping("./data/GNN_Features/002_S_1155_I843510")