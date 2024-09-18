import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig, BertModel
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence

def tokenize_ligand(atom_lines):
    ligand_tokens = []
    for line in atom_lines:
        if line.startswith('HETATM'):
            parts = line.split()
            if len(parts) >= 12:
                atom_type = parts[11].strip()
                try:
                    x, y, z = map(float, parts[6:9])
                    ligand_tokens.append({
                        'atom_type': atom_type,
                        'x': x, 'y': y, 'z': z,
                    })
                except ValueError:
                    continue
    return ligand_tokens

def tokenize_protein_binding_site(atom_lines, ligand_tokens, distance_threshold=5.0):
    binding_site_tokens = []
    
    def calculate_distance(coord1, coord2):
        return np.linalg.norm(np.array(coord1) - np.array(coord2))
    
    for line in atom_lines:
        if line.startswith('ATOM'):
            parts = line.split()
            if len(parts) >= 12:
                residue_type = parts[3].strip()
                atom_type = parts[11].strip()
                try:
                    x, y, z = map(float, parts[6:9])
                    protein_coord = (x, y, z)
                    for ligand in ligand_tokens:
                        ligand_coord = (ligand['x'], ligand['y'], ligand['z'])
                        if calculate_distance(protein_coord, ligand_coord) <= distance_threshold:
                            binding_site_tokens.append({
                                'residue_type': residue_type,
                                'atom_type': atom_type,
                                'x': x, 'y': y, 'z': z
                            })
                            break
                except ValueError:
                    continue
    return binding_site_tokens

def parse_and_tokenize_pdb(pdb_file, distance_threshold=5.0):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    ligand_lines = [line for line in lines if line.startswith('HETATM')]
    protein_lines = [line for line in lines if line.startswith('ATOM')]
    
    ligand_tokens = tokenize_ligand(ligand_lines)
    binding_site_tokens = tokenize_protein_binding_site(protein_lines, ligand_tokens, distance_threshold)
    
    return ligand_tokens, binding_site_tokens

def combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens):
    combined_tokens = [token['atom_type'] for token in ligand_tokens] + [token['residue_type'] for token in binding_site_tokens]
    segment_ids = [0] * len(ligand_tokens) + [1] * len(binding_site_tokens)
    return combined_tokens, segment_ids

def process_pdb_files(directory):
    all_combined_tokens = []
    all_segment_ids = []
    
    for pdb_file in os.listdir(directory):
        if pdb_file.endswith('.pdb'):
            file_path = os.path.join(directory, pdb_file)
            ligand_tokens, binding_site_tokens = parse_and_tokenize_pdb(file_path)
            combined_tokens, segment_ids = combine_ligand_protein_tokens(ligand_tokens, binding_site_tokens)
            all_combined_tokens.append(combined_tokens)
            all_segment_ids.append(segment_ids)
    
    return all_combined_tokens, all_segment_ids


class ProteinLigandDataset(Dataset):
    def __init__(self, input_ids, segment_ids):
        self.input_ids = input_ids
        self.segment_ids = segment_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'token_type_ids': torch.tensor(self.segment_ids[idx], dtype=torch.long),
            'attention_mask': torch.ones(len(self.input_ids[idx]), dtype=torch.long)
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

class ProteinLigandTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_attention_heads=12, num_hidden_layers=12):
        super(ProteinLigandTransformer, self).__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            type_vocab_size=2
        )
        self.bert = BertModel(config)
        self.cls = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        logits = self.cls(sequence_output)
        return logits


def main():
    directory = 'nrsites'
    all_combined_tokens, all_segment_ids = process_pdb_files(directory)

    unique_tokens = list(set([token for tokens in all_combined_tokens for token in tokens]))
    token2id = {token: idx for idx, token in enumerate(unique_tokens)}
    id2token = {idx: token for token, idx in token2id.items()}

    input_ids = [[token2id[token] for token in tokens] for tokens in all_combined_tokens]

    # Split the data into train and validation sets
    train_input_ids, val_input_ids, train_segment_ids, val_segment_ids = train_test_split(
        input_ids, all_segment_ids, test_size=0.2, random_state=42
    )

    train_dataset = ProteinLigandDataset(input_ids=train_input_ids, segment_ids=train_segment_ids)
    val_dataset = ProteinLigandDataset(input_ids=val_input_ids, segment_ids=val_segment_ids)

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
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
            
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
                token_type_ids = batch['token_type_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                
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
