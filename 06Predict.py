import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import xgboost as xgb
import os
import sys
from tqdm import tqdm

# 必要的 PyG 与图形工具
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, coalesce
from torch_cluster import knn_graph
sys.path.append('/mnt/GeneGnn/Github/Data') 
from DeathGRN import DeathGRN
os.chdir("/mnt/GeneGnn/Github/Data")

def discover_new_death_genes(model_path='best_death_gene_model.pth', 
                             data_path='death_gene_graph_data.pt', 
                             metadata_path='gene_metadata_final.csv'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 加载基础图数据 (包含 1280 维 ESM-2 向量)
    print(f"🚀 正在加载 ESM-2 嵌入数据...")
    data = torch.load(data_path, weights_only=False).to(device)
    node_metadata = pd.read_csv(metadata_path)
    num_nodes = data.num_nodes

    # 2. 特征工程重构：确保维度为 1284
    # ---------------------------------------------------------
    # A. 构建异构图 (STRING + KNN)
    print(" 构建异构关系图 (Physical + Semantic)...")
    knn_edge_index = knn_graph(data.x, k=5, loop=False, cosine=True)
    string_edge_index, _ = coalesce(data.edge_index, None, num_nodes)
    
    string_edge_type = torch.zeros(string_edge_index.size(1), dtype=torch.long, device=device)
    knn_edge_type = torch.ones(knn_edge_index.size(1), dtype=torch.long, device=device)

    combined_edge_index = torch.cat([string_edge_index, knn_edge_index], dim=1)
    combined_edge_type = torch.cat([string_edge_type, knn_edge_type], dim=0)

    # B. 计算 3 维拓扑特征 (PageRank, Degree, Clustering)
    print(" 计算全局拓扑特征...")
    nx_graph = to_networkx(Data(edge_index=combined_edge_index, num_nodes=num_nodes), to_undirected=True)
    pr = torch.tensor(list(nx.pagerank(nx_graph).values()), dtype=torch.float32).view(-1, 1).to(device)
    dc = torch.tensor(list(nx.degree_centrality(nx_graph).values()), dtype=torch.float32).view(-1, 1).to(device)
    cc = torch.tensor(list(nx.clustering(nx_graph).values()), dtype=torch.float32).view(-1, 1).to(device)
    topo_features = torch.cat([pr, dc, cc], dim=1) # 3D

    # C. 训练/加载 XGBoost 产生 1 维先验
    print(" 产生 XGBoost 死亡先验评分...")
    labeled_mask = data.y != -1
    X_xgb = torch.cat([data.x, topo_features], dim=1)[labeled_mask].cpu().numpy()
    y_xgb = data.y[labeled_mask].cpu().numpy()
    
    # 使用训练时的超参数
    xgb_predictor = xgb.XGBClassifier(n_estimators=150, max_depth=4, scale_pos_weight=2.45, device='cuda')
    xgb_predictor.fit(X_xgb, y_xgb)
    
    all_features_for_xgb = torch.cat([data.x, topo_features], dim=1).cpu().numpy()
    xgb_probs = xgb_predictor.predict_proba(all_features_for_xgb)[:, 1]
    xgb_feature = torch.tensor(xgb_probs, dtype=torch.float32).view(-1, 1).to(device)

    # D. 最终 1284 维特征拼接
    augmented_x = torch.cat([data.x, topo_features, xgb_feature], dim=1) # 1280+3+1 = 1284
    print(f" 特征空间构建完成，维度: {augmented_x.shape}")

    # 3. 加载深度模型并执行全量推理
    print(f" 正在加载训练好的 DeathGRN 权重: {model_path}...")
    model = DeathGRN(in_channels=1284, hidden_channels=512, out_channels=2, num_layers=1, num_relations=2).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        logits = model(augmented_x, combined_edge_index, combined_edge_type)
        # 经过 Softmax 转化为概率
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # 4. 结果整理与候选基因筛选
    results = node_metadata.copy()
    results['Death_Probability'] = probs
    results['Original_Label'] = data.y.cpu().numpy()
    # 筛选：排除已知标签（正样本和负样本），仅保留 -1 (未标记) 基因
    # 按照死亡概率从高到低排序
    new_candidates = results[results['Original_Label'] == -1].copy()
    new_candidates = new_candidates.sort_values(by='Death_Probability', ascending=False)

    # 5. 保存结果
    output_file = "genome_wide_death_gene_rankings.csv"
    output_file_all = "genome_wide_death_gene_rankings_all.csv"

    new_candidates.to_csv(output_file, index=False)
    results.sort_values(by='Death_Probability', ascending=False).to_csv(output_file_all, index=False)

    print(f"\n✅ 推理完成！已保存全基因组排名至: {output_file}")
    print("\n--- 排名最高的前 15 个潜在新型细胞死亡驱动基因 ---")
    print(new_candidates[['Primary_Symbol', 'Death_Probability']].head(15))
    
    return new_candidates, results

if __name__ == "__main__":
    candidates, results = discover_new_death_genes()