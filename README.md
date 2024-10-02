# Self-Supervised Learning of Protein-Ligand Interactions using BERT  

This repository contains the implementation of a self-supervised Bidirectional Encoder Representations from Transformers (BERT) model designed to understand protein-ligand interactions. The goal is to predict how binding site residues influence ligand behavior using masked language modeling (MLM) tasks for both the ligand and protein binding site tokens.

## 1. Theory

### What are Transformers?

Transformers have revolutionized machine learning, particularly in natural language processing (NLP), by enabling models to handle complex data relationships efficiently. Introduced in the paper *"Attention is All You Need"*, transformers rely on **self-attention mechanisms**, allowing them to capture long-range dependencies in data more effectively than previous architectures like recurrent neural networks (RNNs).

At the core of transformers is the **self-attention** mechanism, which weighs the importance of each part of the input sequence relative to the others. This enables transformers to process input data in parallel rather than sequentially, resulting in faster and more scalable models. Due to their parallelism and ability to capture context across entire sequences, transformers have outperformed older models in tasks like translation, text generation, and understanding.

Transformers are not limited to NLP; they have been applied across various fields such as computer vision and bioinformatics. In fields like protein-ligand interaction modeling, transformers capture the complex spatial and chemical relationships between molecules, making them an invaluable tool for biological and chemical data.

Their flexibility and power make transformers one of the most influential architectures in modern machine learning.

![My image](./images/transformer.png)

### How do Transformer models benefit the study of Protein-Ligand Interactions?
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


### Self-Supervised Learning & Masked Language Model (MLM)

Self-supervised learning (SSL) is ideal when dealing with vast amounts of unlabeled data, as it allows models to generate their own supervision signals from the data itself. In the study of protein-ligand interactions, we have abundant structural data from sources like the Protein Data Bank (PDB), but experimentally derived labels, such as binding affinities, are often limited.

Masked Language Modeling (MLM), a core technique in self-supervised learning, trains a model to predict masked parts of the input based on the surrounding context. In the original NLP setting, MLM teaches transformers like BERT to understand the relationships between words by masking certain words in a sentence.

In the context of protein-ligand interactions, I apply the same principle by masking atoms of the ligand or residues in the binding site. The model learns to predict the masked atoms or residues based on the surrounding structural and chemical context, thereby learning spatial and chemical dependencies critical to protein-ligand interactions.

This method is highly effective for capturing the intricate details of how ligands interact with proteins. Once pre-trained in this way, the model can then be fine-tuned for specific downstream tasks, significantly improving its predictive performance.


## 2. Implementation and Mathametical Foundation of BindAxTransformer

### **Step 1: Extraction of Protein and Ligand Properties**

In this step, we parse Protein Data Bank (PDB) files to extract detailed information about the protein and ligand atoms in a protein-ligand complex. Specifically, we aim to obtain:

- **Atom Type**: The type of each atom (e.g., carbon, nitrogen).

- **Element**: The chemical element symbol (e.g., C for carbon).

- **Coordinates**: The 3D spatial coordinates (x, y, z) of each atom.

- **Residue Information** (for proteins): The amino acid residue type and number.

  
1. **Parsing and Tokenizing Atoms**

```python
def tokenize_atom(line, molecule_type):
# Extract atom details from a line in the PDB file
 ```

* **Input**: A line from the PDB file and a molecule type identifier (0 for ligand, 1 for protein).

* **Process**: Extracts atom properties using string slicing based on PDB format specifications.

* **Output**: A dictionary containing the extracted properties.

2. **Parsing the PDB File**

```python
def parse_and_tokenize_pdb(pdb_file):
# Read the PDB file and tokenize all atoms
```

* **Process**: Reads all lines in the PDB file and separates ligand and protein atoms based on record types (`HETATM` for ligands, `ATOM` for proteins).

* **Output**: A combined list of tokenized ligand and protein atoms.

3. **Calculating Interatomic Distances**

```python 
def calculate_distance(coord1, coord2):
# Compute Euclidean distance between two atoms
```
* **Input**: Coordinates of two atoms.

* **Process**: Calculates the Euclidean distance using the formula:

$$\text{Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$$
  
* **Output**: The scalar distance between the two atoms.

4. **Finding Interactions**

```python
def find_interactions(ligand_tokens, protein_tokens, distance_threshold=5.0):
# Identify atom pairs within a specified distance
```

* **Process**: For each ligand atom, computes distances to all protein atoms and identifies pairs within the distance threshold (e.g., 5.0 Ångströms).

* **Output**: A list of interacting atom pairs along with their distances.


#### **PDB Format Parsing**

- **Atom Line Format**: Each line in a PDB file representing an atom follows a specific format. For example:

```python
ATOM      1  N   MET A   1      38.428  13.104   6.364  1.00 54.69           N
```
- **Columns**:
  - **Atom Serial Number (7-11)**: Unique identifier for the atom.
  - **Atom Name (13-16)**: The name of the atom.
  - **Residue Name (18-20)**: The name of the amino acid residue.
  - **Chain Identifier (22)**: Protein chain identifier.
  - **Residue Sequence Number (23-26)**: The sequence number of the residue.
  - **Coordinates (31-54)**: x, y, z coordinates of the atom.
  - **Element Symbol (77-78)**: The chemical element symbol.

- **Extraction**: Using string slicing, we extract these fields for each atom.

#### **Euclidean Distance Calculation**

Given two atoms with coordinates $$\((x_1, y_1, z_1)\)$$ and $$\((x_2, y_2, z_2)\)$$, the Euclidean distance is calculated as:

$$\text{Distance} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2}$$

This calculation identifies atoms that are spatially close and potentially interacting.

### **Example**

Let's consider a simplified example with one ligand atom and one protein atom.

- **Ligand Atom**:
   - **Atom Type**: C1
   - **Element**: C
   - **Coordinates**: (10.0, 10.0, 10.0)

- **Protein Atom**:
   - **Atom Type**: N
   - **Element**: N
   - **Coordinates**: (12.0, 10.0, 10.0)
   - **Residue Type**: MET
   - **Residue Number**: 1

#### **Distance Calculation**

$$\text{Distance} = \sqrt{(12.0 - 10.0)^2 + (10.0 - 10.0)^2 + (10.0 - 10.0)^2} = \sqrt{(2.0)^2} = 2.0  \text{Å}$$

Since the distance (2.0 Å) is less than the threshold (5.0 Å), we consider these atoms to be interacting.

---

### **Step 2: Tokenization of the Extracted Properties**

In this step, we convert the extracted atom properties into numerical representations (tokens) suitable for input into a BERT model. Tokenization involves mapping discrete categorical variables to integer IDs and preparing continuous variables for model ingestion.

1. **Defining the Dataset Class**

```python

class ProteinLigandDataset(Dataset):
#Custom dataset for handling protein-ligand interactions
```
- **Attributes**:
   - `self.tokens`: A list of interaction tokens extracted previously.
   - `self.token2id`: A mapping from token strings to unique integer IDs.

2. **Implementing `__getitem__`**

```python
def __getitem__(self, idx):
# Retrieves and processes a single data point
```

- **Process**:
   - **Input IDs**: Converts categorical variables (e.g., atom types, elements) to integer IDs using `self.token2id`.
   - **Coordinates**: Combines ligand and protein atom coordinates into a single tensor.
   - **Distance**: The scalar distance between the interacting atoms.
   - **Attention Mask**: A binary mask indicating the presence of tokens (used in BERT models).

3. **Collate Function for DataLoader**

```python
def collate_fn(batch):
# Aggregates multiple data points into a batch
```

- **Process**:
   - Pads sequences to ensure uniform length across the batch.
   - Stacks coordinates and distances into tensors.



4. **Tokenization**

* **Categorical Variables**: Atom types, elements, and residue types are categorical and require encoding.

   * **Token Mapping**:
      * Create a vocabulary (`token2id`) that assigns a unique integer ID to each unique token in the dataset.
* For example:

| Token                     | ID |
|---------------------------|----|
| `ligand_atom_C1`         | 1  |
| `ligand_element_C`       | 2  |
| `protein_atom_N`         | 3  |
| `protein_element_N`      | 4  |
| `protein_residue_MET`    | 5  |


**Input IDs**: Each data point's categorical variables are converted to a sequence of IDs.

**Input IDs** = ID<sub>ligand_atom</sub>, ID<sub>ligand_element</sub>, ID<sub>protein_atom</sub>, ID<sub>protein_element</sub>, ID<sub>protein_residue</sub>


* **Continuous Variables**
   * **Coordinates**: These are continuous variables representing spatial positions.
   * **Normalization**: Coordinates may be normalized to improve model training, typically by centering and scaling:

$$
\text{Normalized } x_i = \frac{x_i - \mu_x}{\sigma_x}
$$

where $$\(\mu_x\)$$ and $$\(\sigma_x\)$$ are the mean and standard deviation of the $$\(x\)$$-coordinates in the dataset.

- **Distance**: Also a continuous variable, possibly normalized similarly.

5. **Attention Mask**

- Used in transformer models to indicate which tokens are valid (1) and which are padding (0).

### **Example**

Using the previous example, let's tokenize the interaction.

- **Tokens**:

  - `ligand_atom_C1`: ID 1

  - `ligand_element_C`: ID 2

  - `protein_atom_N`: ID 3

  - `protein_element_N`: ID 4

  - `protein_residue_MET`: ID 5

- **Input IDs**:

$$\text{Input IDs} = [1, 2, 3, 4, 5]$$

- **Coordinates**:

$$\text{Coordinates} = [10.0, 10.0, 10.0, 12.0, 10.0, 10.0]$$


- **Distance**:

$$\text{Distance} = [2.0]$$

- **Attention Mask**:

$$\text{Attention Mask} = [1, 1, 1, 1, 1]$$


### **Preparing for BERT Model**

The BERT model expects inputs in a specific format:

- **Input IDs**: A sequence of token IDs.

- **Attention Mask**: Indicates which tokens are actual data.

- **Optional Position Embeddings**: Since we're dealing with spatial data, we might include positional information.

#### **Incorporating Coordinates**

- **Embedding Coordinates**:

  - Coordinates can be embedded separately or concatenated with token embeddings.

  - Alternatively, a custom embedding layer can process the coordinates.

#### **Mathematical Representation in the Model**

- **Embedding Layer**:

  - Transforms input IDs into dense vector representations:

  $$\mathbf{E}_{\text{input}} = \text{EmbeddingMatrix} \times \text{Input IDs}$$


- **Model Input**:

  - The model input might be a combination of token embeddings and coordinate embeddings.


**Note**: When implementing the actual model, consider additional preprocessing steps such as handling rare tokens, normalizing continuous variables, and integrating spatial information in a way that leverages the strengths of transformer architectures.
## 3. Detailed Code Explanation for BindAxTransformer

I have created a separate repository that explains the code behind **BindAxTransformer** line by line. You can find that repository [here](https://github.com/gautammalik-git/Understanding-BindAxTransformer.git).


## 4. Usage

### Prerequisites
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
### Running the code
```bash
python bindax_trans.py
```
Ensure that you update the directory name containing the training complexes in the Python script.

## 5. Conclusion 

This repository provides a guide for understanding and implementing a self-supervised transformer model for protein-ligand interactions. Currently, the code includes only coordinates and atom/residue types as tokens. However, you can enhance the model by incorporating additional tokens such as distances between atoms and residues, dihedral angles, charges, and more. The goal of this repository is to offer a flexible foundation for protein-ligand interaction transformer model, allowing anyone to fine-tune it to their specific needs. I'm also working on a larger project that leverages this pre-trained model, but instead of releasing everything at once, I’m sharing this code first. Stay tuned for more exciting updates and keep building your own projects!
