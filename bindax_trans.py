import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
from sklearn.model_selection import train_test_split

def tokenize_atom(line, molecule_type):
    atom_type = line[12:16].strip()
    element = line[76:78].strip()
    x = float(line[30:38].strip())
    y = float(line[38:46].strip())
    z = float(line[46:54].strip())
    residue_name = line[17:20].strip() if molecule_type == 1 else None
    residue_number = int(line[22:26].strip()) if molecule_type == 1 else None
    
    return {
        'molecule_type': molecule_type,
        'atom_type': atom_type,
        'element': element,
        'coordinates': (x, y, z),
        'residue_type': residue_name,
        'residue_number': residue_number
    }

def parse_and_tokenize_pdb(pdb_file):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    ligand_tokens = [tokenize_atom(line, 0) for line in lines if line.startswith('HETATM')]
    protein_tokens = [tokenize_atom(line, 1) for line in lines if line.startswith('ATOM')]
    
    return ligand_tokens + protein_tokens

def calculate_distance(coord1, coord2):
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(coord1, coord2)))

def find_interactions(ligand_tokens, protein_tokens, distance_threshold=5.0):
    interactions = []
    for ligand_atom in ligand_tokens:
        for protein_atom in protein_tokens:
            distance = calculate_distance(ligand_atom['coordinates'], protein_atom['coordinates'])
            if distance <= distance_threshold:
                interactions.append((ligand_atom, protein_atom, distance))
    return interactions

def process_pdb_files(directory):
    all_tokens = []
    
    for pdb_file in os.listdir(directory):
        if pdb_file.endswith('.pdb'):
            file_path = os.path.join(directory, pdb_file)
            tokens = parse_and_tokenize_pdb(file_path)
            
            ligand_tokens = [t for t in tokens if t['molecule_type'] == 0]
            protein_tokens = [t for t in tokens if t['molecule_type'] == 1]
            
            interactions = find_interactions(ligand_tokens, protein_tokens)
            
            for ligand_atom, protein_atom, distance in interactions:
                interaction_token = {
                    'ligand_atom': ligand_atom['atom_type'],
                    'ligand_element': ligand_atom['element'],
                    'ligand_coords': ligand_atom['coordinates'],
                    'protein_atom': protein_atom['atom_type'],
                    'protein_element': protein_atom['element'],
                    'protein_coords': protein_atom['coordinates'],
                    'protein_residue': protein_atom['residue_type'],
                    'protein_residue_num': protein_atom['residue_number'],
                    'distance': distance
                }
                all_tokens.append(interaction_token)
    
    return all_tokens

class ProteinLigandDataset(Dataset):
    def __init__(self, tokens, token2id):
        self.tokens = tokens
        self.token2id = token2id

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token = self.tokens[idx]
        input_ids = [
            self.token2id[f"ligand_atom_{token['ligand_atom']}"],
            self.token2id[f"ligand_element_{token['ligand_element']}"],
            self.token2id[f"protein_atom_{token['protein_atom']}"],
            self.token2id[f"protein_element_{token['protein_element']}"],
            self.token2id[f"protein_residue_{token['protein_residue']}"]
        ]
        coordinates = list(token['ligand_coords']) + list(token['protein_coords'])
        distance = [token['distance']]
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'coordinates': torch.tensor(coordinates, dtype=torch.float),
            'distance': torch.tensor(distance, dtype=torch.float),
            'attention_mask': torch.ones(len(input_ids), dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    coordinates = [item['coordinates'] for item in batch]
    distances = [item['distance'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    coordinates = torch.stack(coordinates)
    distances = torch.stack(distances)
    attention_mask = nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'coordinates': coordinates,
        'distances': distances,
        'attention_mask': attention_mask
    }

class ProteinLigandTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=1152, num_attention_heads=12, num_hidden_layers=12):
        super(ProteinLigandTransformer, self).__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers
        )
        self.bert = BertModel(config)
        self.coordinate_encoder = nn.Linear(6, hidden_size)  # 6 for x,y,z of both ligand and protein
        self.distance_encoder = nn.Linear(1, hidden_size)
        self.cls = nn.Linear(hidden_size, vocab_size)
    def forward(self, input_ids=None, coordinates=None, distances=None, attention_mask=None):
        # Generate token embeddings using BERT embedding layer
        token_embeddings = self.bert.embeddings(input_ids) if input_ids is not None else None
        
        # Coordinates encoding and broadcasting over sequence length
        coordinate_embeddings = self.coordinate_encoder(coordinates).unsqueeze(1)
        
        # Distances encoding and broadcasting over sequence length
        distance_embeddings = self.distance_encoder(distances).unsqueeze(1)
        
        # Ensure embeddings are properly combined with token embeddings
        if token_embeddings is not None:
            embeddings = token_embeddings + coordinate_embeddings + distance_embeddings
        else:
            embeddings = coordinate_embeddings + distance_embeddings
        
        # Pass embeddings to BERT's forward method using inputs_embeds
        outputs = self.bert(inputs_embeds=embeddings, attention_mask=attention_mask)
        sequence_output = outputs[0]  # Output from the last hidden layer

        # Classification step
        logits = self.cls(sequence_output)
        
        return logits

def main():
    directory = 'mysite'
    all_tokens = process_pdb_files(directory)
    
    # Create vocabulary
    vocab = set()
    for token in all_tokens:
        vocab.add(f"ligand_atom_{token['ligand_atom']}")
        vocab.add(f"ligand_element_{token['ligand_element']}")
        vocab.add(f"protein_atom_{token['protein_atom']}")
        vocab.add(f"protein_element_{token['protein_element']}")
        vocab.add(f"protein_residue_{token['protein_residue']}")
    
    token2id = {token: idx for idx, token in enumerate(vocab)}
    id2token = {idx: token for token, idx in token2id.items()}

    # Split the data
    train_tokens, val_tokens = train_test_split(all_tokens, test_size=0.2, random_state=42)

    train_dataset = ProteinLigandDataset(train_tokens, token2id)
    val_dataset = ProteinLigandDataset(val_tokens, token2id)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProteinLigandTransformer(vocab_size=len(token2id)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            input_ids = batch['input_ids'].to(device)
            coordinates = batch['coordinates'].to(device)
            distances = batch['distances'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, coordinates=coordinates, distances=distances, attention_mask=attention_mask)
            
            labels = input_ids.clone()
            mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool()
            labels[~mask] = -100  # Only compute loss on masked tokens

            loss = criterion(outputs.view(-1, len(token2id)), labels.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(device)
                coordinates = batch['coordinates'].to(device)
                distances = batch['distances'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, coordinates=coordinates, distances=distances, attention_mask=attention_mask)
                
                labels = input_ids.clone()
                mask = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool()
                labels[~mask] = -100

                loss = criterion(outputs.view(-1, len(token2id)), labels.view(-1))
                val_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_dataloader):.4f}, Val Loss: {val_loss / len(val_dataloader):.4f}')

    # Save the model
    torch.save(model.state_dict(), 'protein_ligand_model.pth')

if __name__ == "__main__":
    main()
