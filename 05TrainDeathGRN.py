import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, 
    precision_score, confusion_matrix, roc_curve, precision_recall_curve, auc
)
import xgboost as xgb

import pandas as pd
import os
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import numpy as np
import time
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel, SWALR
import sys

import networkx as nx
from torch_geometric.utils import to_networkx, dropout_edge, coalesce

sys.path.append('/mnt/GeneGnn/Data') 
from DeathGRN import FocalLoss, DeathGRN
from torch_cluster import knn_graph

os.chdir("/mnt/GeneGnn/Data")


def weight_init(m):

    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.LayerNorm):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def train_func(data_path='death_gene_graph_data.pt', epochs=600, folds=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    full_data = torch.load(data_path, weights_only=False).to(device)
    
    labeled_mask = full_data.y != -1
    labeled_idx = torch.where(labeled_mask)[0].cpu().numpy()
    labels = full_data.y[labeled_idx].cpu().numpy()
    
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)    
    
    all_fold_metrics = []
    all_fold_losses = [] 
    
    tprs = []
    precisions_list = []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)

    best_auprc = 0.0
    best_model_path = 'best_death_gene_model.pth'

    print(f" Original STRING number of edges: {full_data.edge_index.size(1)}")

    knn_edge_index = knn_graph(full_data.x, k=5, loop=False, cosine=True)
    print(f" New KNN semantically connected edge number: {knn_edge_index.size(1)}")
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # 【 Heterogeneous graph reconstruction 】 : generate edge_type and retain multiple biological relationships
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # 2. Remove weights separately and make sure they are clean inside
    string_edge_index, _ = coalesce(full_data.edge_index, None, full_data.num_nodes)
    knn_edge_index, _ = coalesce(knn_edge_index, None, full_data.num_nodes)

    # 3. Create relational tags (0 for physical interaction, 1 for sequence semantics)
    string_edge_type = torch.zeros(string_edge_index.size(1), dtype=torch.long, device=device)
    knn_edge_type = torch.ones(knn_edge_index.size(1), dtype=torch.long, device=device)

    # 4. Concatenation into a Multi-Relational Graph
    combined_edge_index = torch.cat([string_edge_index, knn_edge_index], dim=1)
    combined_edge_type = torch.cat([string_edge_type, knn_edge_type], dim=0)

    full_data.edge_index = combined_edge_index
    full_data.edge_type = combined_edge_type 
    full_data.edge_attr = None
    print(f" Heterogeneous graph is built! Total number of edges: {combined_edge_index.size(1)}")    

    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # [New core module: Extracting graph topology features]
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    print(" Computing global network topology features (PageRank, Degree, Clustering)..." )

    # Temporarily convert PyG data format to NetworkX for computation
    temp_data = Data(edge_index=full_data.edge_index, num_nodes=full_data.num_nodes)
    nx_graph = to_networkx(temp_data, to_undirected=True)
    
    # PageRank
    pr_dict = nx.pagerank(nx_graph)
    pr_values = [pr_dict[i] for i in range(full_data.num_nodes)]
    
    # Degree Centrality
    dc_dict = nx.degree_centrality(nx_graph)
    dc_values = [dc_dict[i] for i in range(full_data.num_nodes)]
    
    # Clustering Coefficient
    cc_dict = nx.clustering(nx_graph)
    cc_values = [cc_dict[i] for i in range(full_data.num_nodes)]
    
    topo_features = torch.tensor([pr_values, dc_values, cc_values], dtype=torch.float32).T.to(device)
    full_data.x = torch.cat([full_data.x, topo_features], dim=1)
    
    print(f" Topological feature injection complete! The current base feature dimension is upgraded to: {full_data.x.spape [1]} dimension ")
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    print(f" Device validation: {torch.cuda.get_device_name(0)} | Ready to start 10%hierarchical validation \n")

    fold_pbar = tqdm(enumerate(skf.split(labeled_idx, labels)), total=folds, desc="Overall Progress", unit="fold")

    for fold, (train_idx, test_idx) in fold_pbar:
        print(f"\n========== Fold {fold + 1} ==========")
    
        # -- -- -- -- -- -- -- -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 【 XGBoost Feature Injection 】
        # -- -- -- -- -- -- -- -- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # 1. Extract the data of the current Fold
        X_train_xgb = full_data.x[train_idx].cpu().numpy()
        y_train_xgb = full_data.y[train_idx].cpu().numpy()
        
        valid_mask = (y_train_xgb != -1)
        X_train_xgb_clean = X_train_xgb[valid_mask]
        y_train_xgb_clean = y_train_xgb[valid_mask]
        # =================================================
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.5,
            scale_pos_weight=2.45, 
            random_state=42,
            tree_method='hist',  
            device='cuda'
        )
        
        xgb_model.fit(X_train_xgb_clean, y_train_xgb_clean)
        
        all_xgb_probs = xgb_model.predict_proba(full_data.x.cpu().numpy())[:, 1]
        
        xgb_feature = torch.tensor(all_xgb_probs, dtype=torch.float32).unsqueeze(1).to(device)
        
        augmented_x = torch.cat([full_data.x.to(device), xgb_feature], dim=1)
        # ---------------------------------------------------------

        model = RelationalAssassinGNN(in_channels=1284, hidden_channels=256, out_channels=2, num_layers=1, num_relations=2).to(device)
        model.apply(weight_init)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        swa_model = AveragedModel(model)
        # SWA 
        swa_scheduler = SWALR(optimizer, swa_lr=0.0005) 
        swa_start = int(epochs * 0.75)

        criterion = FocalLoss(
                                alpha=1.0, 
                                gamma=2.0, 
                                weight=torch.tensor([1.0, 2.4]).to(device) 
                            )

        t_mask = torch.zeros(full_data.num_nodes, dtype=torch.bool, device=device)
        v_mask = torch.zeros(full_data.num_nodes, dtype=torch.bool, device=device)
        t_mask[labeled_idx[train_idx]] = True
        v_mask[labeled_idx[test_idx]] = True
        
        current_fold_loss = []
        epoch_pbar = tqdm(range(1, epochs + 1), desc=f"Fold {fold+1} Training", leave=False)
        
        model.train()
        for epoch in epoch_pbar:
            optimizer.zero_grad()

            out = model(augmented_x, full_data.edge_index, full_data.edge_type)
            loss = criterion(out[t_mask], full_data.y[t_mask])
            loss.backward()
            optimizer.step()

            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                pass
            
            loss_val = loss.item()
            current_fold_loss.append(loss_val)
            epoch_pbar.set_postfix({"loss": f"{loss_val:.4f}"})

        all_fold_losses.append(current_fold_loss)

        swa_model.eval()

        with torch.no_grad():
            logits = swa_model(augmented_x, full_data.edge_index, full_data.edge_type)
            probs = F.softmax(logits[v_mask], dim=1)[:, 1].cpu().numpy()
            preds = logits[v_mask].argmax(dim=1).cpu().numpy()
            true_y = full_data.y[v_mask].cpu().numpy()
            
            auroc = roc_auc_score(true_y, probs)
            auprc = average_precision_score(true_y, probs)
            f1 = f1_score(true_y, preds)
            tn, fp, fn, tp = confusion_matrix(true_y, preds).ravel()
            
            # ROC
            fpr_f, tpr_f, _ = roc_curve(true_y, probs)
            tprs.append(np.interp(mean_fpr, fpr_f, tpr_f))
            tprs[-1][0] = 0.0
            
            # PR
            prec_f, rec_f, _ = precision_recall_curve(true_y, probs)
            precisions_list.append(np.interp(mean_recall, rec_f[::-1], prec_f[::-1]))
            # --------------------------------------

            if auprc > best_auprc:
                best_auprc = auprc
                torch.save(swa_model.module.state_dict(), best_model_path)

            all_fold_metrics.append({
                'Fold': fold + 1, 'AUROC': auroc, 'AUPRC': auprc, 'F1': f1,
                'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn
            })
            
            fold_pbar.set_postfix({"Last_AUPRC": f"{auprc:.3f}"})
    
    # Generate a Loss graph
    plt.figure(figsize=(10, 5))
    for i, losses in enumerate(all_fold_losses):
        plt.plot(losses, alpha=0.3, label=f'Fold {i+1}' if i < 3 else "")
    avg_loss = np.mean(all_fold_losses, axis=0)
    plt.plot(avg_loss, color='black', linewidth=2, label='Mean Loss')
    plt.title('10-Fold Training Loss Convergence')
    plt.xlabel('Epochs'); plt.ylabel('Loss')
    plt.legend(); plt.grid(True, alpha=0.5)
    plt.savefig('combined_loss_curve.png', dpi=300)
    
    #
    loss_csv = pd.DataFrame(all_fold_losses).T
    loss_csv.columns = [f'Fold_{i+1}' for i in range(folds)]
    loss_csv['Mean_Loss'] = avg_loss
    loss_csv.to_csv('loss_curve_data.csv', index_label='Epoch')

    # Generate ROC curve
    plt.figure(figsize=(8, 8))
    for i, tpr_fold in enumerate(tprs):
        plt.plot(mean_fpr, tpr_fold, alpha=0.3, lw=1)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color='blue', label=f'Mean ROC (AUC = {auc(mean_fpr, mean_tpr):.4f})', lw=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title('10-Fold ROC Curves'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right'); plt.grid(True, alpha=0.3)
    plt.savefig('combined_roc_curve.png', dpi=300)

    roc_csv = pd.DataFrame(np.array(tprs).T, columns=[f'Fold_{i+1}_TPR' for i in range(folds)])
    roc_csv.insert(0, 'FPR', mean_fpr)
    roc_csv['Mean_TPR'] = mean_tpr
    roc_csv.to_csv('roc_curve_data.csv', index=False)

    # --- Generate PR graph ---
    plt.figure(figsize=(8, 8))
    for i, prec_fold in enumerate(precisions_list):
        plt.plot(mean_recall, prec_fold, alpha=0.3, lw=1)
    mean_precision = np.mean(precisions_list, axis=0)
    plt.plot(mean_recall, mean_precision, color='green', label=f'Mean PR (AUPRC = {np.mean([m["AUPRC"] for m in all_fold_metrics]):.4f})', lw=2)
    plt.title('10-Fold PR Curves'); plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.legend(loc='lower left'); plt.grid(True, alpha=0.3)
    plt.savefig('combined_pr_curve.png', dpi=300)

    pr_csv = pd.DataFrame(np.array(precisions_list).T, columns=[f'Fold_{i+1}_Precision' for i in range(folds)])
    pr_csv.insert(0, 'Recall', mean_recall)
    pr_csv['Mean_Precision'] = mean_precision
    pr_csv.to_csv('pr_curve_data.csv', index=False)

    df = pd.DataFrame(all_fold_metrics)
    print("\n" + "═"*60)
    print(f"{'Fold':^6} | {'AUROC':^8} | {'AUPRC':^8} | {'F1':^8} | {'Correct (TP+TN)':^15}")
    print("─"*60)
    for _, row in df.iterrows():
        correct = int(row['TP'] + row['TN'])
        print(f"{int(row['Fold']):^6} | {row['AUROC']:^8.4f} | {row['AUPRC']:^8.4f} | {row['F1']:^8.4f} | {correct:^15}")
    print("═"*60)
    
    summary = df[['AUROC', 'AUPRC', 'F1']].mean()
    Print (f "average performance indicators: AUROC: {summary: [' AUROC]. 4 f} | AUPRC: {summary: [' AUPRC]. 4 f} | F1: {summary: [' F1 ']. 4 f}")

    df.to_csv(f'{folds}fold_results_summary.csv', index=False)
    print(f"\n Graphs and raw data CSV have been saved to the current directory." )

if __name__ == "__main__":
    train_func(epochs=600, folds=10)