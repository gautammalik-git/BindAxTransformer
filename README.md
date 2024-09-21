# Self-Supervised Learning of Protein-Ligand Interactions using BERT  

This repository contains the implementation of a self-supervised Bidirectional Encoder Representations from Transformers (BERT) model designed to understand protein-ligand interactions. The goal is to predict how binding site residues influence ligand behavior using masked language modeling (MLM) tasks for both the ligand and protein binding site tokens.

## 1. Theory

### a. How do Transformer models benefit the study of Protein-Ligand Interactions?
In biological systems, proteins interact with ligands in binding sites, influencing processes like enzyme catalysis, signal transduction, and drug binding. Understanding how specific residues in the binding site affect ligand binding can significantly aid in drug discovery and design. 

Transformer models offer significant advantages in the study of Protein-Ligand Interactions, primarily due to their ability to leverage large datasets and capture complex relationships in biological data.

1. **Utilization of Extensive Datasets**:
   We often have access to a wealth of experimentally derived protein-ligand complexes from sources like the Protein Data Bank (PDB). This data is invaluable for training machine learning models to predict various outcomes, such as binding site conformations, ligand binding activities, and binding affinities. However, specific applications, like predicting the binding affinity of a ligand for a particular target (e.g., the beta-adrenergic receptor), can suffer from sparse data. 

2. **Role of Pre-trained Models**:
   This is where pre-trained transformer models come into play. By training on a comprehensive dataset these models learn the fundamental principles of protein-ligand interactions. They become adept at understanding the underlying biological mechanisms without being tailored to a specific protein or ligand.

3. **Generalization Across Different Systems**:
   Since transformers are capable of generalizing from their training data, we can leverage a pre-trained model to then fine-tune it for a specific target, such as the beta-adrenergic receptor. This approach is advantageous because it allows us to start with a robust foundation of learned interactions, requiring fewer specific data points to achieve meaningful predictions. Fine-tuning on the specific target can lead to improved accuracy, as the model already understands the broader context of protein-ligand interactions and can focus on the unique characteristics of the target in question.

It is for these reasons that I have developed a self-supervised transformer-based model utilizing the 65,000 protein-ligand complexes from the PDB, aimed at advancing our understanding of protein-ligand interactions and improving predictive accuracy in this area of study.

![My image](./images/pre-trained.png)

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

## 3. Reproducing the Code

### a. Environment Setup
To run the code, you will need:
- Python 3.7+
- PyTorch
- Hugging Face `transformers`
- scikit-learn for splitting data

You can install the required dependencies using:
```bash
pip install torch transformers scikit-learn numpy
```

If in case the torch module is still not installed try using:

```bash
pip3 install torch torchvision 
```

If you are using conda you can also try:

```bash
conda install pytorch torchvision -c pytorch
```

## 4. Usage
```bash
python bindax_trans.py
```
Ensure that you update the directory name containing the training complexes in the Python script.

## 5. Conclusion 

This repository provides a guide for understanding and implementing a self-supervised transformer model for protein-ligand interactions. Currently, the code includes only coordinates and atom/residue types as tokens. However, you can enhance the model by incorporating additional tokens such as distances between atoms and residues, dihedral angles, charges, and more. The goal of this repository is to offer a flexible foundation for protein-ligand interaction transformer model, allowing anyone to fine-tune it to their specific needs. I'm also working on a larger project that leverages this pre-trained model, but instead of releasing everything at once, I’m sharing this code first. Stay tuned for more exciting updates and keep building your own projects!
