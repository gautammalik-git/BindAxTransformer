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

### Protein-Ligand Complex Tokenization

This repository explains the process of tokenizing protein-ligand complexes, combining ligand and protein binding site atom information, padding for uniform token size, and converting them into embeddings using a BERT-based tokenizer. This method is applied in self-supervised learning models for understanding protein-ligand interactions.

### **Step 1: Tokenization of Protein-Ligand Complexes**

### **1.1 Extraction of Atoms and Coordinates**

For each protein-ligand complex, we extract the following information:
- **Ligand**: A list of atoms and their corresponding 3D coordinates $$\(x, y, z)\$$.
- **Protein Binding Site**: A list of residues, each containing multiple atoms and their 3D coordinates $$\(x, y, z)\$$.

#### **Notation:**
- Ligand atoms: $$\( L_i \)$$ represents the $$\(ith\)$$ atom of the ligand.
- Protein binding site atoms: $$\( P_j^r \)$$, where $$\(P_j\)$$ represents the $$\(j\)-th$$ atom in residue $$\(r\)$$.

#### **Example Complexes:**

- **Complex 1:**
  - Ligand: $$\( [L_1, L_2, L_3, L_4] \)$$
  - Binding Site: $$\( [P_1^1, P_2^1, P_1^2, \ldots, P_3^6] \)$$
  
- **Complex 2:**
  - Ligand: $$\( [L_1, L_2, L_3] \)$$
  - Binding Site: $$\( [P_1^1, P_1^2, \ldots, P_2^5] \)$$
  
- **Complex 3:**
  - Ligand: $$\( [L_1, L_2, L_3, L_4, L_5] \)$$
  - Binding Site: $$\( [P_1^1, P_2^1, \ldots, P_4^7] \)$$

#### **Extracted Tuple Format:**
For each atom, we extract a tuple: $$\( (atom\_type, x, y, z) \)$$.

### **1.2 Combination of Tokens**

For each complex, the extracted atoms are combined into a sequence of tokens consisting of ligand atoms and binding site atoms. The token list for each complex can be represented as:

$$\
T = [L_1, L_2, \dots, L_m, P_1, P_2, \dots, P_n]
\$$

Where:
- $$\( L_1, \dots, L_m \)$$ are ligand atoms,
- $$\( P_1, \dots, P_n \)$$ are protein atoms from binding site residues.

Each token $$\( T_i \)$$ is represented as $$\( T_i = (atom\_type_i, x_i, y_i, z_i) \)$$.

### **1.3 Padding for Uniform Token Size**

Since the number of atoms varies across complexes, we pad the token lists to a fixed length, $$\( N_{\text{max}} \)$$, which is the maximum number of tokens in any complex.

For each complex:
- **Complex 1**: $$\( |T_1| = 10 \)$$, pad with $$\( N_{\text{max}} - 10 \)$$.
- **Complex 2**: $$\( |T_2| = 8 \)$$, pad with $$\( N_{\text{max}} - 8 \)$$.
- **Complex 3**: $$\( |T_3| = 12 \$$), no padding needed.

### **1.4 Token to Embedding Conversion**

Next, the tokens are converted into embeddings using the BERT tokenizer. Each token $$\( T_i \)$$ is mapped to a vector in the embedding space. Let the embedding dimension be $$\( d \)$$, typically 768 for BERT.

Each complex $$\( C_k \)$$ is represented as a matrix of embeddings $$\( E_k \)$$:

$$E_k = [e_T1  e_T2  e_T3 .... e_TNmax]\$$ 

Where $$\( \mathbf{e}_{T_i} \)$$ is the embedding of token $$\( T_i \)$$.

#### **Example Embedding Dimensions:**
- **Complex 1**: $$\( E_1 \in \mathbb{R}^{N_{\text{max}} \times 768} \)$$
- **Complex 2**: $$\( E_2 \in \mathbb{R}^{N_{\text{max}} \times 768} \)$$
- **Complex 3**: $$\( E_3 \in \mathbb{R}^{N_{\text{max}} \times 768} \)$$


### 1.5 Single Token Embedding for Ligand Atom

Given a token $$\( T_i \)$$ that represents a single ligand atom, the embedding process maps the token to a high-dimensional vector. The features of this token typically include:
1. **Atom Type**: Symbol of the atom (e.g., "C" for Carbon, "O" for Oxygen).
2. **3D Coordinates**: $$\( (x_i, y_i, z_i) \)$$ representing the spatial location of the atom.
3. **Charge**: The atom’s charge or other atomic properties.
4. **Positional Encoding**: The atom’s position in the sequence of tokens.

For a ligand atom with atom type "C" (carbon), 3D coordinates $$\( (1.24, -2.55, 0.67) \)$$, and charge 0, its embedding vector is composed of these different components:

```math
\mathbf{e}_{T_i} = \mathbf{v}_\text{atom\_type} + \mathbf{v}_\text{coordinates} + \mathbf{v}_\text{charge} + \mathbf{v}_\text{positional\_encoding}
```

Let’s break this down:

1. **Atom Type Embedding** $$\( \mathbf{v}_\text{atom\_type} \)$$:  
   This encodes the type of atom. For a carbon atom, this embedding might be:
   
$$v_atom_type = [0.1, 0.2, 0.05, ..., 0.3] in ℝ^d$$

2. **3D Coordinates Embedding** $$\( \mathbf{v}_\text{coordinates} \)$$:  
   The coordinates $$\( (1.24, -2.55, 0.67) \)$$ are encoded into a vector:
   
   ```math
   \mathbf{v}_\text{coordinates} = \text{MLP}([1.24, -2.55, 0.67]) = [0.05, 0.12, 0.03, \dots, 0.08] \in \mathbb{R}^{d}
   ```

3. **Charge Embedding** \( \mathbf{v}_\text{charge} \):  
   The charge of the atom (here 0) is also embedded:

   ```math
   \mathbf{v}_\text{charge} = [0.0, 0.0, 0.0, \dots, 0.0] \in \mathbb{R}^{d}
   ```

5. **Positional Encoding** $$\( \mathbf{v}_\text{positional\_encoding} \)$$:  
   The positional encoding captures the position of this atom in the sequence. For the $$\( i \)-th$$ atom, the positional encoding might be:
   
   $$
   \mathbf{v}_\text{positional\_encoding} = [0.15, 0.03, 0.07, \dots, 0.01] \in \mathbb{R}^{d}
   $$

### Final Embedding for Single Token $$\( T_i \)$$

Summing all these components, we get the final embedding vector for token $$\( T_i \)$$:

$$
\mathbf{e}_{T_i} = [0.1, 0.2, 0.05, \dots, 0.3] + [0.05, 0.12, 0.03, \dots, 0.08] + [0.0, 0.0, 0.0, \dots, 0.0] + [0.15, 0.03, 0.07, \dots, 0.01]
$$

This results in a high-dimensional vector $$\( \mathbf{e}_{T_i} \in \mathbb{R}^{768} \)$$, which contains information about the atom's type, its spatial location, charge, and its position in the sequence. This vector is then part of the larger matrix representing the entire complex.

### Single Token Embedding Representation

For a single ligand token $$\( T_i \)$$ (a carbon atom), the embedding looks like this:

$$
\mathbf{e}_{T_i} = \text{BERTTokenizer}(T_i) \in \mathbb{R}^{768}
$$

This embedding will be one of the rows in the final matrix $$\( E_k \)$$, where $$\( k \)$$ represents the index of the complex. Each row corresponds to one token (atom) from either the ligand or the protein binding site.





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
