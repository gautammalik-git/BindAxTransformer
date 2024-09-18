# Protein-Ligand Interaction Self-Supervised Transformer Model

This repository contains the implementation of a self-supervised transformer model designed to understand protein-ligand interactions. The goal is to predict how binding site residues influence ligand behavior using masked language modeling (MLM) tasks for both the ligand and protein binding site tokens.

## 1. Theory

### a. Protein-Ligand Interactions
In biological systems, proteins interact with ligands in binding sites, influencing processes like enzyme catalysis, signal transduction, and drug binding. Understanding how specific residues in the binding site affect ligand binding can significantly aid in drug discovery and design.

### b. Transformers for Protein-Ligand Interactions
Transformers, traditionally used in NLP, excel at learning context. For protein-ligand interactions, transformers can be trained using masked token prediction to learn relationships between atoms in the ligand and residues in the binding site. This approach is useful because:

- **Self-supervised learning** doesn't require explicit labels, making it ideal for large, unlabeled datasets.
- **Masked Language Modeling (MLM)** allows the transformer to predict masked atoms or residues based on the surrounding context, making the model sensitive to local interactions.

### c. Objective
The aim is to build a transformer model that masks a portion of the input atoms (for ligands) and residues (for proteins) and predicts them based on the rest of the sequence. This method helps the model understand spatial and chemical relationships in protein-ligand complexes.

## 2. Code Explanation

### a. Tokenizing Ligand and Binding Site
The code begins by parsing Protein Data Bank (PDB) files to extract ligand and protein binding site coordinates. These coordinates are then tokenized into atom types and residue types:
- **`tokenize_ligand()`** extracts atom type and 3D coordinates from ligand atoms.
- **`tokenize_protein_binding_site()`** extracts residue and atom types from the binding site residues that are within a distance threshold from the ligand atoms.

### b. Combining Tokens
Once tokenized, ligand and binding site tokens are combined into a single sequence, representing the protein-ligand complex:
- **`combine_ligand_protein_tokens()`** joins the ligand and binding site tokens, assigning segment IDs to distinguish between ligand and protein tokens.

### c. Preparing the Dataset
The combined tokens from all PDB files in a directory are processed:
- **`process_pdb_files()`** tokenizes all the PDB files and returns tokenized sequences for further training.
- A dataset is created using **PyTorch’s Dataset class**, allowing the tokens to be fed into the transformer model efficiently.

### d. Transformer Model
The transformer is built using the **Hugging Face BERT architecture**:
- **`ProteinLigandTransformer`** defines a model based on BERT with customized configuration (12 hidden layers, 12 attention heads). The input to the model is the combined ligand and protein tokens.
- The model predicts the masked tokens by reconstructing atom and residue types based on context.

### e. Training Loop
The training loop implements the masked language model training:
- **Masking Strategy**: 15% of the input tokens (atoms and residues) are randomly masked.
- **Training**: The model learns to predict these masked tokens, minimizing the cross-entropy loss between the predicted and actual tokens.

### f. Saving the Model
After training, the model’s parameters are saved to a file for future use in predicting novel protein-ligand interactions.

## 3. Tips to Reproduce the Code

### a. Environment Setup
To run the code, you will need:
- Python 3.7+
- PyTorch
- Hugging Face `transformers`
- scikit-learn for splitting data

You can install the required dependencies using:
```bash
pip install torch transformers scikit-learn numpy
