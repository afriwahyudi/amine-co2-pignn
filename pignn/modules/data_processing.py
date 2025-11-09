import os
import pandas as pd
from rdkit import Chem
import torch
from torch_geometric.data import Data
from collections import Counter

def load_dataset(path):
    df = pd.read_csv(path, delimiter=';')
    return df

def smiles_to_graph(smiles, name):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        mol = Chem.AddHs(mol)
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

    atom_features = []
    edge_index = []
    edge_attr = []

    # Define atom types for one-hot encoding (common elements)
    atomic_types = ['H', 'C', 'N', 'O']

    # Atom features (Node features)
    nitrogen_classifications = []
    for atom in mol.GetAtoms():
        atom_type = atom.GetSymbol()
        
        # track nitrogen types globally
        if atom_type == 'N':
            num_c_neighbors = len([nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C'])

            if num_c_neighbors == 1:
                nitrogen_classifications.append('primary')
            elif num_c_neighbors == 2:
                nitrogen_classifications.append('secondary')
            elif num_c_neighbors == 3:
                nitrogen_classifications.append('tertiary')
            else:
                nitrogen_classifications.append('other')

        # 1. One-hot encoding for atomic types (H, C, N, O) -> size 4
        atomic_vector = [1 if atom_type == el else 0 for el in atomic_types]

        # 2. One-hot encoding for hybridization (sp, sp2, sp3) -> size 3
        hybridization = atom.GetHybridization()
        if hybridization == Chem.rdchem.HybridizationType.SP3:
            hybridization_vector = [1, 0, 0]
        elif hybridization == Chem.rdchem.HybridizationType.SP2:
            hybridization_vector = [0, 1, 0]
        elif hybridization == Chem.rdchem.HybridizationType.SP:
            hybridization_vector = [0, 0, 1]
        else:
            hybridization_vector = [0, 0, 0]

        # 3. Binary aromaticity -> size 1
        aromaticity = [1 if atom.GetIsAromatic() else 0] 

        # 4. Binary feature for ring membership -> size 1
        ring_feature = [atom.IsInRing()]
        
        # 5. Nitrogen type identifier -> size 3
        if atom_type == 'N':
            num_c_neighbors = len([nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C'])
            
            if num_c_neighbors == 1:
                nitrogen_type = [1, 0, 0]
            elif num_c_neighbors == 2:
                nitrogen_type = [0, 1, 0]
            elif num_c_neighbors == 3:
                nitrogen_type = [0, 0, 1]
            else:
                nitrogen_type = [0, 0, 0]
        else:
            nitrogen_type = [0, 0, 0]

        # 6. Carbon type identifier -> size 4
        if atom_type == 'C':
            num_c_neighbors = len([nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C'])
            
            if num_c_neighbors == 1:
                carbon_type = [1, 0, 0, 0]
            elif num_c_neighbors == 2:
                carbon_type = [0, 1, 0, 0]
            elif num_c_neighbors == 3:
                carbon_type = [0, 0, 1, 0]
            elif num_c_neighbors == 4:
                carbon_type = [0, 0, 0, 1]
            else:
                carbon_type = [0, 0, 0, 0]
        else:
            carbon_type = [0, 0, 0, 0]
        
        # 7. Hydrogen donor identifier -> size 1
        hydrogen_donor = [1] if any((atom_type == el and bond.GetOtherAtom(atom).GetSymbol() == 'H') 
                                for bond in atom.GetBonds() 
                                for el in ['N', 'O']) else [0]

        # 8. Hydrogen acceptor identifier -> size 1
        hydrogen_acceptor = [1] if atom_type in ['N', 'O'] else [0]
        hydrogen_donor_feature = hydrogen_donor
        hydrogen_acceptor_feature = hydrogen_acceptor 

        # 9. Hydrogen in reactive sites identifier -> size 1
        hydrogen_in_reactive_site = [0]
        if atom_type == 'H':
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetSymbol() == 'N':
                    hydrogen_in_reactive_site = [1]
                    break
        else:
            hydrogen_in_reactive_site = [0]

        # 10. Hydroxyl membership identifier -> size 1
        hydroxyl_membership = [0]
        if atom_type == 'O':
            hydrogen_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
            if len(hydrogen_neighbors) == 1:
                carbon_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'C']
                if len(carbon_neighbors) == 1:
                    carbon_atom = carbon_neighbors[0]
                    is_carbonyl = any(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and 
                                    bond.GetOtherAtom(carbon_atom).GetSymbol() == 'O' 
                                    for bond in carbon_atom.GetBonds())
                    if not is_carbonyl:
                        hydroxyl_membership = [1]
        elif atom_type == 'H':
            oxygen_neighbors = [nbr for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'O']
            if len(oxygen_neighbors) == 1:
                oxygen_atom = oxygen_neighbors[0]
                carbon_neighbors = [nbr for nbr in oxygen_atom.GetNeighbors() if nbr.GetSymbol() == 'C']
                if len(carbon_neighbors) == 1:
                    carbon_atom = carbon_neighbors[0]
                    is_carbonyl = any(bond.GetBondType() == Chem.rdchem.BondType.DOUBLE and 
                                    bond.GetOtherAtom(carbon_atom).GetSymbol() == 'O' 
                                    for bond in carbon_atom.GetBonds())
                    if not is_carbonyl:
                        hydroxyl_membership = [1]
                            
        atom_feature = (atomic_vector + hybridization_vector + 
                        aromaticity + ring_feature + nitrogen_type +
                        carbon_type +
                        hydrogen_donor_feature +
                        hydrogen_acceptor_feature +
                        hydrogen_in_reactive_site +
                        hydroxyl_membership)
        
        atom_features.append(atom_feature)  
        

    # Bond features (Edge connections)
    bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
    for bond in mol.GetBonds():
        src, dst = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.append([src, dst])
        edge_index.append([dst, src])

        bond_type_vector = [1 if bond.GetBondType() == b_type else 0 for b_type in bond_types]
        
        edge_attr.append(bond_type_vector)
        edge_attr.append(bond_type_vector)

    # Convert to PyTorch tensors
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    nitrogen_counts = Counter([t for t in nitrogen_classifications if t in ['primary', 'secondary', 'tertiary']])

    if not nitrogen_counts:
        graph_type = "none"
    else:
        type_parts = []
        for n_type in ['primary', 'secondary', 'tertiary']:
            count = nitrogen_counts.get(n_type, 0)
            if count > 0:
                if count == 1:
                    type_parts.append(n_type)
                else:
                    type_parts.append(f"{count}-{n_type}")
        
        graph_type = '-'.join(type_parts)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.name = name
    data.type = graph_type
    data.mol = mol
    data.primary_count = nitrogen_counts.get('primary', 0)
    data.secondary_count = nitrogen_counts.get('secondary', 0) 
    data.tertiary_count = nitrogen_counts.get('tertiary', 0)

    return data

def process_dataset(df, smiles_dict, use_component_mapping=True, system_col='System', 
                    smiles_col='SMILES', conc_col='Conc (mol/L)', temp_col='T (K)', 
                    pco2_col='PCO2 (kPa)', aco2_col='aCO2 (mol CO2/mol amine)', 
                    doi_col = 'DOI', ref_col = 'Reference'):
    reverse_smiles_dict = {v: k for k, v in smiles_dict.items()}

    graphs = []
    for idx, row in df.iterrows():
        system_val = row.get(system_col)
        if pd.isna(system_val) or system_val == '':
            print(f"Warning: Skipping row {idx} because '{system_col}' is missing.")
            continue
        if use_component_mapping and system_val in smiles_dict:
            smiles = smiles_dict[system_val]
            final_name = system_val
        elif smiles_col in row and pd.notna(row[smiles_col]):
            candidate_smiles = row[smiles_col]
            if use_component_mapping and candidate_smiles in reverse_smiles_dict:
                final_name = reverse_smiles_dict[candidate_smiles]
            else:
                final_name = system_val
            smiles = candidate_smiles
        else:
            print(f"Warning: No SMILES found for {system_val}")
            continue

        graph = smiles_to_graph(smiles, final_name)
        if graph:
            graph.conc = row.get(conc_col, None)
            graph.temp = row.get(temp_col, None)
            graph.pco2 = row.get(pco2_col, None)
            graph.aco2 = row.get(aco2_col, None)
            graph.ref  = row.get(ref_col , None)
            graph.DOI  = row.get(doi_col , None)
            graphs.append(graph)
    return graphs

def generate_graphs(df):
    smiles_col = None
    name_col = None
    
    for col in df.columns:
        if 'smiles' in col.lower():
            smiles_col = col
        elif 'molecule' in col.lower() or 'system' in col.lower() or 'mol_id' in col.lower():
            name_col = col
    
    if smiles_col is None or name_col is None:
        raise ValueError("DataFrame does not contain the required columns for SMILES or Molecule name.")
    
    graphs = []
    seen_molecules = set()
    
    for idx, row in df.iterrows():
        system_val = row.get(name_col)
        
        if pd.isna(system_val) or system_val == '':
            print(f"Warning: Skipping row {idx} because '{name_col}' is missing.")
            continue
        smiles = row.get(smiles_col)
        
        if pd.isna(smiles):
            print(f"Warning: No SMILES found for {system_val}")
            continue
        if smiles in seen_molecules:
            continue
        seen_molecules.add(smiles)
        graph = smiles_to_graph(smiles, system_val)
        if graph:
            graphs.append(graph)
    
    return graphs

