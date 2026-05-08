import torch
from transformers import AutoTokenizer, EsmModel
import pandas as pd
import numpy as np
from tqdm import tqdm

import os 
os.chdir("/mnt/GeneGnn/data")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/esm2_t33_650M_UR50D" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmModel.from_pretrained(model_name)
model = model.half().to(device)
model.eval()

df = pd.read_csv("gene_seq_string_map.csv")

def get_embedding(sequence):
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().float().numpy()


print("Start extracting embeddings...")
embeddings = []
for seq in tqdm(df['Sequence']):
    embeddings.append(get_embedding(seq))

embedding_matrix = np.array(embeddings)
np.save("gene_esm2_features.npy", embedding_matrix)
df[['Primary_Symbol', 'STRING']].to_csv("gene_metadata_final.csv", index=False)
print("Feature extraction complete! The results have been saved as.npy files.")