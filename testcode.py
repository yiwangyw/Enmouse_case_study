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
    def __init__(self, tensors, labels, batch_size):
        self.tensors = tensors
        self.labels = labels
        self.batch_size = batch_size
        
        # 分离正负样本 (0为正样本，1为负样本)
        self.pos_indices = (labels == 0).nonzero(as_tuple=True)[0]
        self.neg_indices = (labels == 1).nonzero(as_tuple=True)[0]
        
        print(f"正样本数量: {len(self.pos_indices)}")
        print(f"负样本数量: {len(self.neg_indices)}")
        
        # 计算正负样本比例
        total_samples = len(labels)
        self.pos_ratio = len(self.pos_indices) / total_samples
        self.neg_ratio = len(self.neg_indices) / total_samples
        
        # 根据比例计算每个batch中的正负样本数
        self.pos_samples_per_batch = int(self.batch_size * self.pos_ratio)
        self.neg_samples_per_batch = self.batch_size - self.pos_samples_per_batch
        
        print(f"每个batch中的正样本数: {self.pos_samples_per_batch}")
        print(f"每个batch中的负样本数: {self.neg_samples_per_batch}")
        
        # 计算可以构建的完整batch数量
        self.num_batches = min(
            len(self.pos_indices) // self.pos_samples_per_batch,
            len(self.neg_indices) // self.neg_samples_per_batch
        )
        
        print(f"可构建的完整batch数量: {self.num_batches}")
        
        if self.num_batches == 0:
            raise ValueError(f"无法创建batch。正样本: {len(self.pos_indices)}, "
                           f"负样本: {len(self.neg_indices)}, "
                           f"每个batch所需正样本: {self.pos_samples_per_batch}, "
                           f"每个batch所需负样本: {self.neg_samples_per_batch}")

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError("Index out of bounds")
        
        # 为当前batch选择正负样本
        start_pos = idx * self.pos_samples_per_batch
        start_neg = idx * self.neg_samples_per_batch
        
        # 获取当前batch的正负样本索引
        batch_pos_indices = self.pos_indices[start_pos:start_pos + self.pos_samples_per_batch]
        batch_neg_indices = self.neg_indices[start_neg:start_neg + self.neg_samples_per_batch]
        
        # 合并并打乱索引
        batch_indices = torch.cat([batch_pos_indices, batch_neg_indices])
        batch_indices = batch_indices[torch.randperm(len(batch_indices))]
        
        return self.tensors[batch_indices], self.labels[batch_indices]

def create_balanced_dataloader(tensors, labels, batch_size):
    """创建保持原始比例的数据加载器"""
    try:
        dataset = BalancedBatchDataset(tensors, labels, batch_size)
        loader = DataLoader(dataset, batch_size=None, shuffle=False)
        print(f"数据加载器批次数: {len(loader)}")
        return loader
    except Exception as e:
        raise ValueError(f"创建数据加载器失败: {str(e)}")

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
        tensors = []
        for tensor, label in test_dataset:
            tensor = ensure_cpu_tensor(tensor)
            label = ensure_cpu_tensor(label)
            tensors.append((tensor, label))
        X = torch.stack([t[0] for t in tensors])
        y = torch.stack([t[1] for t in tensors])
        return X, y
    except Exception as e:
        raise

def load_new_test_data(num_samples_to_insert):
    """加载新的测试数据"""
    try:
        file_path = os.path.join(Config.get_data_dir(), 
                                f'predict_samples_user{Config.USER_ID}_{Config.WINDOW_SIZE}.csv')
        X_insert_all = np.loadtxt(file_path, delimiter=',', skiprows=1)
        
        if len(X_insert_all) < num_samples_to_insert:
            raise ValueError(f"需要 {num_samples_to_insert} 个新样本，但只有 {len(X_insert_all)} 个可用")
        
        X_insert = X_insert_all[:num_samples_to_insert]
        X_insert = torch.from_numpy(X_insert).to(torch.float32)
        X_insert = X_insert.unsqueeze(dim=1)
        label_insert = torch.ones(num_samples_to_insert).to(torch.int64)
        
        return X_insert, label_insert
    except Exception as e:
        raise



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
        raise

def evaluate_model(model, data_loader):
    """评估模型性能"""
    if len(data_loader) == 0:
        raise ValueError("数据加载器为空，无法进行评估")
        
    batch_metrics = []
    model.eval()
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            scores = probs[:, 1]
            preds = torch.argmax(probs, dim=1)
            
            batch_metrics.append({
                'preds': preds.cpu().numpy(),
                'scores': scores.cpu().numpy(),
                'labels': labels.cpu().numpy()
            })
            
    if not batch_metrics:
        raise ValueError("没有处理任何batch数据")
        
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
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores)
    auc = metrics.auc(fpr, tpr)
    
    # 计算EER
    fnr = 1 - tpr
    # 找到FPR和FNR最接近的点
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    EER = fpr[np.nanargmin(np.absolute(fnr - fpr))]
    
    return precision, recall, f1, accuracy, auc, fpr, tpr, EER

def plot_combined_metrics(original_results, new_results):
    """绘制组合指标可视化"""
    metrics_names = ['Recall', 'Accuracy', 'Precision', 'F1', 'AUC', 'EER']
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
        ensure_directories()
        X_orig, y_orig = load_test_data()
        
        if X_orig.shape[0] == 0 or y_orig.shape[0] == 0:
            raise ValueError("原始数据集为空")

        num_neg_samples = int((y_orig == 1).sum().item())
        X_insert, label_insert = load_new_test_data(num_neg_samples)
        
        if X_insert.shape[0] == 0 or label_insert.shape[0] == 0:
            raise ValueError("新插入数据集为空")
        
        X_combined = torch.cat([X_orig, X_insert])
        y_combined = torch.cat([y_orig, label_insert])
        
        print("\n 创建数据加载器...")
        try:
            print("创建原始数据加载器...")
            orig_test_loader = create_balanced_dataloader(X_orig, y_orig, Config.BATCH_SIZE)
            
            print("\n创建合并数据加载器...")
            combined_test_loader = create_balanced_dataloader(X_combined, y_combined, Config.BATCH_SIZE)
        except Exception as e:
            raise ValueError(f"创建数据加载器失败: {str(e)}")
        
        shape_single_mouse_traj = X_orig.shape[2]
        model = MouseNeuralNetwork(shape_single_mouse_traj).to(Config.DEVICE)
        model_path = os.path.join(Config.BASE_PATH, 
                                f'pt/gru-only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
        checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        
        pred_ids_orig, scores_orig, labels_orig = evaluate_model(model, orig_test_loader)
        metrics_orig = compute_metrics(pred_ids_orig, scores_orig, labels_orig)
        precision_orig, recall_orig, f1_orig, acc_orig, auc_orig, fpr_orig, tpr_orig, eer_orig = metrics_orig
        
        pred_ids_new, scores_new, labels_new = evaluate_model(model, combined_test_loader)
        metrics_new = compute_metrics(pred_ids_new, scores_new, labels_new)
        precision_new, recall_new, f1_new, acc_new, auc_new, fpr_new, tpr_new, eer_new = metrics_new
        
        original_results = {
            'user_id': f'user{Config.USER_ID}',
            'window_size': Config.WINDOW_SIZE,
            'recall': recall_orig,
            'accuracy': acc_orig,
            'precision': precision_orig,
            'f1': f1_orig,
            'auc': auc_orig,
            'eer': eer_orig
        }
        
        new_results = {
            'user_id': f'user{Config.USER_ID}',
            'window_size': Config.WINDOW_SIZE,
            'recall': recall_new,
            'accuracy': acc_new,
            'precision': precision_new,
            'f1': f1_new,
            'auc': auc_new,
            'eer': eer_new
        }
        
        save_results_to_csv(original_results, 'original_test')
        save_results_to_csv(new_results, 'new_test')
        
        plot_combined_roc(
            (fpr_orig, tpr_orig, auc_orig),
            (fpr_new, tpr_new, auc_new)
        )
        plot_combined_metrics(original_results, new_results)
        
        print("\nMetric       Original Test    New Test")
        print("-" * 45)
        metrics = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC', 'EER']
        for metric in metrics:
            orig_value = original_results[metric.lower()]
            new_value = new_results[metric.lower()]
            print(f"{metric:<12} {orig_value:>8.4f}        {new_value:>8.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        print("详细错误信息:")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_testing()