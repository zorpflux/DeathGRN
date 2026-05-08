import os
import pandas as pd
from collections import Counter

import os
os.chdir("/mnt/GeneGnn/data")

def build_seed_gene_set(data_dir):
    """
    All txt gene list files in the specified directory were read, merged to remove duplicates, and the number of supporting evidence was counted.
    """
    all_genes = []
    file_count = 0
    
    print(f"To start scanning the directory: {data_dir}...")
    
    for filename in os.listdir(data_dir):
        if filename.endswith(".grp") or filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            file_count += 1
            
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    gene = line.strip()
                    # Simple cleaning: Skip empty lines, URL links, or header lines
                    if gene and not gene.startswith('http') and len(gene) < 20:
                        all_genes.append(gene.upper())
                        
    print(f"A total of {file_count} files were read, and {len(all_genes)} (including duplicates) gene records were extracted.")
    
    # Frequency of gene occurrence (finding consensus genes)
    gene_counter = Counter(all_genes)
    
    df_seeds = pd.DataFrame(gene_counter.items(), columns=['Gene_Symbol', 'Evidence_Count'])
    
    # They were arranged in descending order according to the number of evidences (frequency of occurrence)
    df_seeds = df_seeds.sort_values(by='Evidence_Count', ascending=False).reset_index(drop=True)
    
    print(f"A total of {len(df_seeds)} unique candidate genes for cell death were obtained.")
    
    return df_seeds

def build_negative_seed_set(neg_dir, positive_seeds_csv):
    """
    1. Read all candidate negative sample files
    2. Read the existing positive sample file
    3. The overlapping genes were strictly excluded to generate absolute negative sample sets
    """
    pos_df = pd.read_csv(positive_seeds_csv)
    positive_genes = set(pos_df['Gene_Symbol'].astype(str).str.upper())
    print(f"Loaded positive sample genes: {len(positive_genes)}")

    raw_neg_genes = []
    for filename in os.listdir(neg_dir):
        if filename.endswith(".txt") or filename.endswith(".grp"):
            file_path = os.path.join(neg_dir, filename)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    gene = line.strip()
                    if gene and not gene.startswith('http') and len(gene) < 20:
                        raw_neg_genes.append(gene.upper())

    unique_neg_candidates = set(raw_neg_genes)
    print(f"Extract negative sample candidates: {len(unique_neg_candidates)}")

    absolute_neg_genes = unique_neg_candidates - positive_genes
    
    print(f"After deleting genes overlapping with positive samples, the absolute negative samples were {len(absolute_neg_genes)}")

    neg_df = pd.DataFrame(list(absolute_neg_genes), columns=['Gene_Symbol'])
    neg_df['Label'] = 0
    neg_df.to_csv("cell_death_negative_seeds.csv", index=False)
    
    return neg_df

if __name__ == '__main__':

    data_folder = "/mnt/GeneGnn/data/CellDeathGRP" 
    
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    else:
        final_seeds_df = build_seed_gene_set(data_folder)
        
        # Core strategy: Filtering and exporting
        # Strategy A: Cast as wide a net as possible for pan-cell death genes (as long as they appear once)
        final_seeds_df.to_csv("cell_death_positive_seeds_ALL.csv", index=False)

        # Strategy B: If you want to extract the core genes (present in at least two different lists)
        core_seeds_df = final_seeds_df[final_seeds_df['Evidence_Count'] >= 2]
        core_seeds_df.to_csv("cell_death_positive_seeds_CORE.csv", index=False)
        
        print("\n=== Extract preview (top 10 genes with highest consensus) ===")
        print(final_seeds_df.head(10))
        print("\nProcessing done!")

    neg_folder = "/mnt/GeneGnn/data/NegGeneGRP"
    pos_file = "/mnt/GeneGnn/data/cell_death_positive_seeds_ALL.csv"

    if not os.path.exists(neg_folder):
        os.makedirs(neg_folder)
    else:
        final_neg_df = build_negative_seed_set(neg_folder, pos_file)
        print("\n-- Negative sample preview --")
        print(final_neg_df.head(10))
    

