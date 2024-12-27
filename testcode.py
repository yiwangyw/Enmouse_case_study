import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn import metrics
import pandas as pd
import pickle
import os
import sys
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from config import Config

# Add model path
sys.path.append(Config.get_model_path())
from resnet_mouse import _ResNetLayer
from mouse_traj_classification import MouseNeuralNetwork
from data_utils import read_test_data_shape, ensure_cpu_tensor

class BalancedBatchDataset(Dataset):
    def __init__(self, tensors, labels, batch_size, pos_ratio=None):
        self.tensors = tensors
        self.labels = labels
        self.batch_size = batch_size
        
        # 分离正负样本
        pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
        
        # 如果没有指定正样本比例，使用原始数据中的比例
        if pos_ratio is None:
            pos_ratio = len(pos_indices) / len(labels)
        
        # 计算每个batch中的正负样本数量
        self.pos_per_batch = int(batch_size * pos_ratio)
        self.neg_per_batch = batch_size - self.pos_per_batch
        
        # 打乱正负样本索引
        self.pos_indices = pos_indices[torch.randperm(len(pos_indices))]
        self.neg_indices = neg_indices[torch.randperm(len(neg_indices))]
        
        # 计算完整batch的数量
        self.num_batches = min(
            len(pos_indices) // self.pos_per_batch,
            len(neg_indices) // self.neg_per_batch
        )

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError("Index out of bounds")
        
        # 获取当前batch的正负样本索引
        pos_start = idx * self.pos_per_batch
        neg_start = idx * self.neg_per_batch
        
        batch_pos_indices = self.pos_indices[pos_start:pos_start + self.pos_per_batch]
        batch_neg_indices = self.neg_indices[neg_start:neg_start + self.neg_per_batch]
        
        # 合并并打乱当前batch的索引
        batch_indices = torch.cat([batch_pos_indices, batch_neg_indices])
        batch_indices = batch_indices[torch.randperm(len(batch_indices))]
        
        # 返回batch数据
        return self.tensors[batch_indices], self.labels[batch_indices]

def create_balanced_dataloader(tensors, labels, batch_size, pos_ratio=None):
    dataset = BalancedBatchDataset(tensors, labels, batch_size, pos_ratio)
    return DataLoader(dataset, batch_size=None, shuffle=False)

def ensure_directories():
    """确保所需目录存在"""
    directories = [
        Config.get_results_path(),
        Config.get_figures_path()
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def load_test_data():
    """加载测试数据集"""
    test_path = os.path.join(Config.get_data_dir(), 'X_test_loader.pkl')
    try:
        with open(test_path, 'rb') as f:
            test_dataset = pickle.load(f)
            
        # 确保数据集中的张量都在CPU上
        tensors = []
        for tensor, label in test_dataset:
            tensor = ensure_cpu_tensor(tensor)
            label = ensure_cpu_tensor(label)
            tensors.append((tensor, label))
            
        # 创建新的数据集
        X = torch.stack([t[0] for t in tensors])
        y = torch.stack([t[1] for t in tensors])
        
        # 计算原始正样本比例
        pos_ratio = (y == 1).float().mean().item()
        
        return create_balanced_dataloader(X, y, Config.BATCH_SIZE, pos_ratio)
    except Exception as e:
        print(f"Error loading test dataset: {str(e)}")
        raise

def load_new_test_data():
    """加载新的测试数据"""
    try:
        file_path = os.path.join(Config.get_data_dir(), 
                                f'predict_samples_user{Config.USER_ID}_{Config.WINDOW_SIZE}.csv')
        X_insert = np.loadtxt(file_path, delimiter=',', skiprows=1)
        X_insert = torch.from_numpy(X_insert).to(torch.float32)  # 先转换为CPU张量
        X_insert = X_insert.unsqueeze(dim=1)
        label_insert = torch.zeros((len(X_insert))).to(torch.int64)  # 先创建在CPU上的张量
        return X_insert, label_insert
    except Exception as e:
        print(f"Error loading new test data: {str(e)}")
        raise

def insert_new_test_data(test_loader, X_insert, label_insert):
    """插入新的测试数据并创建平衡的dataloader"""
    # 获取原始数据
    X_orig = torch.cat([batch[0] for batch in test_loader])
    y_orig = torch.cat([batch[1] for batch in test_loader])
    
    # 合并原始数据和新数据
    X_combined = torch.cat([X_orig, X_insert])
    y_combined = torch.cat([y_orig, label_insert])
    
    # 计算新的正样本比例
    pos_ratio = (y_combined == 1).float().mean().item()
    
    return create_balanced_dataloader(X_combined, y_combined, Config.BATCH_SIZE, pos_ratio)

def save_results_to_csv(results_dict, test_type):
    """保存结果到CSV文件"""
    filename = os.path.join(Config.get_results_path(), 
                           f'{test_type}_results_user{Config.USER_ID}.csv')
    new_row = pd.DataFrame([results_dict])
    
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        df.to_csv(filename, index=False)
    except Exception as e:
        print(f"Error saving results to CSV: {str(e)}")
        raise

def evaluate_model(model, data_loader):
    """评估模型性能"""
    batch_metrics = []
    total_inference_time = 0
    num_inferences = 0

    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            start_time = time.time()
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1]
            end_time = time.time()
            
            total_inference_time += (end_time - start_time)
            num_inferences += inputs.size(0)
            
            preds = torch.argmax(probs, dim=1)
            
            # 计算当前batch的指标
            batch_metrics.append({
                'preds': preds.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'labels': labels.cpu().numpy()
            })
            
    avg_inference_time = total_inference_time / num_inferences
    print(f"Average inference time: {avg_inference_time:.6f} seconds")
    
    # 合并所有batch的结果
    all_preds = np.concatenate([m['preds'] for m in batch_metrics])
    all_scores = np.concatenate([m['scores'] for m in batch_metrics])
    all_labels = np.concatenate([m['labels'] for m in batch_metrics])
    
    return all_preds, all_scores, all_labels

def compute_metrics(pred_ids, scores, labels):
    """计算评估指标"""
    precision = precision_score(labels, pred_ids)
    recall = recall_score(labels, pred_ids)
    f1 = f1_score(labels, pred_ids)
    accuracy = accuracy_score(labels, pred_ids)
    
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    
    return precision, recall, f1, accuracy, auc, fpr, tpr

def plot_combined_metrics(original_results, new_results):
    """绘制组合指标可视化"""
    metrics_names = ['Recall', 'Accuracy', 'Precision', 'F1', 'AUC']
    original_values = [original_results[k.lower()] for k in metrics_names]
    new_values = [new_results[k.lower()] for k in metrics_names]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_names))
    width = 0.35
    
    plt.bar(x - width/2, original_values, width, label='Original Test', color='skyblue')
    plt.bar(x + width/2, new_values, width, label='New Test', color='lightcoral')
    
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    plt.title(f'Metrics Comparison (User {Config.USER_ID}, Window {Config.WINDOW_SIZE})')
    plt.xticks(x, metrics_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(Config.get_figures_path(), 
                            f'combined_metrics_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    plt.savefig(save_path)
    plt.close()

def plot_combined_roc(original_data, new_data):
    """绘制组合ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    fpr_orig, tpr_orig, auc_orig = original_data
    plt.plot(fpr_orig, tpr_orig, 'b-', 
             label=f'Original Test (AUC = {auc_orig:.3f})',
             linewidth=2)
    
    fpr_new, tpr_new, auc_new = new_data
    plt.plot(fpr_new, tpr_new, 'r--',
             label=f'New Test (AUC = {auc_new:.3f})',
             linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k:', label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves Comparison (User {Config.USER_ID}, Window {Config.WINDOW_SIZE})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(Config.get_figures_path(), 
                            f'combined_roc_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    plt.savefig(save_path)
    plt.close()

def run_testing():
    """主测试函数"""
    try:
        # 确保目录存在
        ensure_directories()
        
        # 加载测试数据
        X_test_loader = load_test_data()
        
        # 加载新测试数据
        X_insert, label_insert = load_new_test_data()
        
        # 获取数据形状
        shape_single_mouse_traj = read_test_data_shape(X_test_loader)
        new_test_dataloader = insert_new_test_data(X_test_loader, X_insert, label_insert)
        
        # 初始化和加载模型
        model = MouseNeuralNetwork(shape_single_mouse_traj[2]).to(Config.DEVICE)
        model_path = os.path.join(Config.BASE_PATH, 
                                f'数据处理代码/gru-only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
        checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        
        # 评估原始测试集
        print("\nEvaluating original test set:")
        pred_ids_orig, scores_orig, labels_orig = evaluate_model(model, X_test_loader)
        metrics_orig = compute_metrics(pred_ids_orig, scores_orig, labels_orig)
        precision_orig, recall_orig, f1_orig, acc_orig, auc_orig, fpr_orig, tpr_orig = metrics_orig
        
        # 评估新测试集
        print("\nEvaluating new test set:")
        pred_ids_new, scores_new, labels_new = evaluate_model(model, new_test_dataloader)
        metrics_new = compute_metrics(pred_ids_new, scores_new, labels_new)
        precision_new, recall_new, f1_new, acc_new, auc_new, fpr_new, tpr_new = metrics_new
        
        # 准备结果
        original_results = {
            'user_id': f'user{Config.USER_ID}',
            'window_size': Config.WINDOW_SIZE,
            'recall': recall_orig,
            'accuracy': acc_orig,
            'precision': precision_orig,
            'f1': f1_orig,
            'auc': auc_orig
        }
        
        new_results = {
            'user_id': f'user{Config.USER_ID}',
            'window_size': Config.WINDOW_SIZE,
            'recall': recall_new,
            'accuracy': acc_new,
            'precision': precision_new,
            'f1': f1_new,
            'auc': auc_new
        }
        
        # 保存结果
        save_results_to_csv(original_results, 'original_test')
        save_results_to_csv(new_results, 'new_test')
        
        # 绘制结果
        plot_combined_roc(
            (fpr_orig, tpr_orig, auc_orig),
            (fpr_new, tpr_new, auc_new)
        )
        plot_combined_metrics(original_results, new_results)
        
        # 打印结果比较
        print("\nResults Comparison:")
        print("Metric       Original Test    New Test")
        print("-" * 45)
        metrics = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC']
        for metric in metrics:
            orig_value = original_results[metric.lower()]
            new_value = new_results[metric.lower()]
            print(f"{metric:<12} {orig_value:>8.4f}        {new_value:>8.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error in testing: {str(e)}")
        raise

if __name__ == "__main__":
    run_testing()