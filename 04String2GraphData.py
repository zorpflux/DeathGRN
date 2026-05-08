import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

import os 
os.chdir("/mnt/GeneGnn/data")

# ==========================================
# 1. Load previous metadata (make sure nodes are in the same order)
# ==========================================
node_metadata = pd.read_csv("gene_metadata_final.csv")
# ESM-2 feature vector [number of nodes, 1280]
x_features = np.load("gene_esm2_features.npy")

# STRING_ID (ENSP) -> Matrix row index (0 to N-1)
string_to_idx = {string_id: i for i, string_id in enumerate(node_metadata['STRING'])}
mapping_ser = pd.Series(node_metadata.index.values, index=node_metadata['STRING'])

# ==========================================
# 2. Handle STRING network files (Links)
# ==========================================
links_df = pd.read_csv("STRING/9606.protein.links.v12.0.txt.gz", sep=" ")

# Filter low confidence edges (threshold of 700)
links_df = links_df[links_df['combined_score'] >= 700].reset_index(drop=True)

# Convert ENSP ID to an integer index
# Some edge-linked proteins may not be in the Swiss-Prot list and need to be removed
def map_to_index(row):
    p1, p2 = row['protein1'], row['protein2']
    if p1 in string_to_idx and p2 in string_to_idx:
        return string_to_idx[p1], string_to_idx[p2]
    return None

mapped_edges = links_df.apply(map_to_index, axis=1)
mapped_edges = mapped_edges.dropna()

# Edge indexes and weights are extracted
edge_index_list = list(zip(*mapped_edges))
edge_index = torch.tensor(edge_index_list, dtype=torch.long)
# Normalized weight (0-1)
edge_attr = torch.tensor(links_df.loc[mapped_edges.index, 'combined_score'].values / 1000.0, dtype=torch.float).view(-1, 1)

# ==========================================
# 3. Inject seed Labels
# ==========================================
pos_seeds = pd.read_csv("cell_death_positive_seeds_ALL.csv")['Gene_Symbol'].tolist()
neg_seeds = pd.read_csv("cell_death_negative_seeds.csv")['Gene_Symbol'].tolist()

num_nodes = len(node_metadata)
y = torch.full((num_nodes,), -1, dtype=torch.long) 
train_mask = torch.zeros(num_nodes, dtype=torch.bool)

symbol_to_idx = dict(zip(node_metadata['Primary_Symbol'], range(num_nodes)))

for gene in pos_seeds:
    if gene in symbol_to_idx:
        idx = symbol_to_idx[gene]
        y[idx] = 1
        train_mask[idx] = True

for gene in neg_seeds:
    if gene in symbol_to_idx:
        idx = symbol_to_idx[gene]
        y[idx] = 0
        train_mask[idx] = True


# ==========================================
# 4. Wrapped as a PyG Data object
# ==========================================
data = Data(
    x=torch.tensor(x_features, dtype=torch.float),
    edge_index=edge_index,
    edge_attr=edge_attr,
    y=y,
    train_mask=train_mask
)

torch.save(data, 'death_gene_graph_data.pt')
print(f"The graph data is built!" )
print(f"number of nodes: {data.num_nodes}, number of edges: {data.num_edges}")
print(f"training set dimension scale: {train_mask. Sum (). The item (s)/num_nodes:. 2%}")