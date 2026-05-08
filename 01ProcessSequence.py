import pandas as pd
from Bio import SeqIO

import os

os.chdir("/mnt/GeneGnn/data")

df = pd.read_csv("uniprotkb_AND_model_organism_9606_AND_r_2026_04_22.tsv", sep="\t") # from Uniport

df = df.dropna(subset=['STRING'])
# Extract the main Gene Symbol (Gene Names column usually contains more than one, take the first one)
df['Primary_Symbol'] = df['Gene Names'].str.split(' ').str[0]

# Handling redundancy: If a gene corresponds to more than one protein, the one with the longest length is retained
df = df.sort_values(by='Length', ascending=False).drop_duplicates('Primary_Symbol')

fasta_sequences = {}
for record in SeqIO.parse("uniprotkb_AND_model_organism_9606_AND_r_2026_04_22.fasta", "fasta"): # from Uniport
    accession = record.id.split('|')[1]
    fasta_sequences[accession] = str(record.seq)

# The sequences were merged into the table
df['Sequence'] = df['Entry'].map(fasta_sequences)
df = df.dropna(subset=['Sequence']) # Records without sequences were eliminated

df[['Primary_Symbol', 'Entry', 'STRING', 'Sequence']].to_csv("gene_seq_string_map.csv", index=False)
print(f"finish! ")
