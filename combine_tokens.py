from Bio.PDB import PDBParser
import numpy as np
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
import random
from transformers import BertTokenizer, BertForMaskedLM, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset, DataLoader

# Function to extract ligand tokens from HETATM lines
def tokenize_ligand(atom_lines):
    ligand_tokens = []
    for line in atom_lines:
        if line.startswith('HETATM'):
            parts = line.split()
            if len(parts) >= 12:
                atom_type = parts[11].strip()  # Atom type (e.g., C, O)
                try:
                    x = float(parts[6].strip())   # X coordinate
                    y = float(parts[7].strip())   # Y coordinate
                    z = float(parts[8].strip())   # Z coordinate
                except ValueError:
                    continue  # Skip lines where coordinates cannot be parsed
                
                token = {
                    'atom_type': atom_type,
                    'x': x,
                    'y': y,
                    'z': z,
                }
                ligand_tokens.append(token)
    return ligand_tokens

# Function to extract protein binding site tokens from ATOM lines
def tokenize_protein_binding_site(atom_lines, ligand_tokens, distance_threshold=5.0):
    binding_site_tokens = []
    
    def calculate_distance(coord1, coord2):
        return np.linalg.norm(np.array(coord1) - np.array(coord2))
    
    for line in atom_lines:
        if line.startswith('ATOM'):
            parts = line.split()
            if len(parts) >= 12:
                residue_type = parts[3].strip()  # Residue type (e.g., ALA, SER)
                atom_type = parts[11].strip()    # Atom type (e.g., C, N)
                try:
                    x = float(parts[6].strip())     # X coordinate
                    y = float(parts[7].strip())     # Y coordinate
                    z = float(parts[8].strip())     # Z coordinate
                except ValueError:
                    continue  # Skip lines where coordinates cannot be parsed
                
                # Check if this protein residue is within distance threshold from any ligand atom
                protein_coord = (x, y, z)
                for ligand in ligand_tokens:
                    ligand_coord = (ligand['x'], ligand['y'], ligand['z'])
                    if calculate_distance(protein_coord, ligand_coord) <= distance_threshold:
                        token = {
                            'residue_type': residue_type,
                            'atom_type': atom_type,
                            'x': x,
                            'y': y,
                            'z': z
                        }
                        binding_site_tokens.append(token)
                        break  # Avoid adding the same residue multiple times
    
    return binding_site_tokens

def parse_and_tokenize_pdb(pdb_file, distance_threshold=5.0):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    # Separate lines into ligand and protein atom lines
    ligand_lines = [line for line in lines if line.startswith('HETATM')]
    protein_lines = [line for line in lines if line.startswith('ATOM')]
    
    # Tokenize ligand and protein
    ligand_tokens = tokenize_ligand(ligand_lines)
    binding_site_tokens = tokenize_protein_binding_site(protein_lines, ligand_tokens, distance_threshold)
    
    return ligand_tokens, binding_site_tokens



#################
directory = 'nrsites'

for pdb_file in os.listdir(directory):
    if pdb_file.endswith('.pdb'):
        file_path = os.path.join(directory, pdb_file)
        ligand_tokens, binding_site_tokens = parse_and_tokenize_pdb(file_path, distance_threshold=5.0)
   

# Combine the ligand and protein tokens

def combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens):
    combined_tokens = []
    segment_ids = []  # Segment ids to differentiate ligand and protein (0 for ligand, 1 for protein)
    
    # Combine ligand tokens
    for token in ligand_tokens:
        combined_tokens.append(token['atom_type'])
        segment_ids.append(0)  # 0 for ligand tokens
    
    # Combine protein tokens
    for token in binding_site_tokens:
        combined_tokens.append(token['residue_type'])
        segment_ids.append(1)  # 1 for protein tokens
    
    return combined_tokens, segment_ids
 
combined_tokens, segment_ids = combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens)


# List to store combined tokens and segment IDs for all complexes
all_combined_tokens = []
all_segment_ids = []


# Process each PDB file in the directory
for pdb_file in os.listdir(directory):
    if pdb_file.endswith('.pdb'):
        file_path = os.path.join(directory, pdb_file)
        ligand_tokens, binding_site_tokens = parse_and_tokenize_pdb(file_path, distance_threshold=5.0)
        
        # Combine the ligand and protein tokens for each complex
        combined_tokens, segment_ids = combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens)
        
        # Accumulate the tokens and segment IDs
        all_combined_tokens.extend(combined_tokens)
        all_segment_ids.extend(segment_ids)

# Now, all_combined_tokens and all_segment_ids contain data for all PDB files in 'nrsites'
print("All Combined Tokens:", all_combined_tokens)
print("All Segment IDs:", all_segment_ids)
   
