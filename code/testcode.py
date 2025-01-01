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
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc
from config import Config
from tqdm import tqdm

# Add model path
sys.path.append(Config.get_model_path())
from resnet_mouse import _ResNetLayer
from mouse_traj_classification import MouseNeuralNetwork

def inspect_dataset(dataset):
    """检查数据集的格式和内容"""
    print(f"\nDataset type: {type(dataset)}")
    print(f"Dataset length: {len(dataset)}")
    
    try:
        first_item = dataset[0]
        print(f"First item type: {type(first_item)}")
        if isinstance(first_item, (tuple, list)):
            print(f"First item length: {len(first_item)}")
            for i, component in enumerate(first_item):
                print(f"Component {i} type: {type(component)}")
                print(f"Component {i} shape: {component.shape if hasattr(component, 'shape') else 'No shape'}")
        else:
            print(f"First item shape: {first_item.shape if hasattr(first_item, 'shape') else 'No shape'}")
    except Exception as e:
        print(f"Error inspecting first item: {str(e)}")

def compute_eer(fpr, tpr, thresholds):
    """计算EER值，增加错误处理和边界检查"""
    try:
        # 检查输入数据的有效性
        if len(fpr) < 2 or len(tpr) < 2:
            print("警告: FPR或TPR数据点不足")
            return 0.0, thresholds[0] if len(thresholds) > 0 else 0.0

        fnr = 1 - tpr
        absolute_diff = np.abs(fpr - fnr)
        
        # 检查是否所有值都是NaN
        if np.all(np.isnan(absolute_diff)):
            print("警告: EER计算中出现全NaN值")
            return 0.0, thresholds[0] if len(thresholds) > 0 else 0.0
            
        idx = np.nanargmin(absolute_diff)
        
        # 基本EER计算
        eer = (fpr[idx] + fnr[idx]) / 2
        
        # 验证计算结果
        if np.isnan(eer) or np.isinf(eer):
            print("警告: EER计算结果无效")
            return 0.0, thresholds[idx]
            
        return eer, thresholds[idx]
        
    except Exception as e:
        print(f"EER计算出错: {str(e)}")
        return 0.0, thresholds[0] if len(thresholds) > 0 else 0.0

def create_evaluation_dataloader(tensors, labels, batch_size):
    """创建评估用的数据加载器"""
    dataset = TensorDataset(tensors, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

def evaluate_model(model, data_loader, threshold=0.5):
    """评估模型性能，增加数据验证"""
    model.eval()
    all_scores = []
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            outputs = model(inputs)
            if outputs.size(1) == 2:
                scores = torch.softmax(outputs, dim=1)[:, 1]
            else:
                scores = torch.sigmoid(outputs.squeeze())
            
            # 确保scores在有效范围内
            scores = torch.clamp(scores, min=0.0, max=1.0)
            preds = (scores > threshold).float()
            
            # 转换为numpy并验证数值
            scores_np = scores.cpu().numpy()
            if np.any(np.isnan(scores_np)):
                print("警告：检测到NaN预测分数")
                scores_np = np.nan_to_num(scores_np, 0.5)
                
            all_scores.extend(scores_np)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    # 最后的验证
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 验证数据的有效性
    if np.any(np.isnan(all_scores)):
        print("警告：最终分数中存在NaN值")
        all_scores = np.nan_to_num(all_scores, 0.5)
    
    return all_preds, all_scores, all_labels

def compute_metrics(pred_ids, scores, labels):
    """计算评估指标"""
    try:
        # 验证输入数据
        if len(np.unique(labels)) < 2:
            print("警告: 标签中只包含单一类别，无法计算某些指标")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 1.0 if np.all(pred_ids == labels) else 0.0,
                'auc': 0.5,
                'eer': 0.5,
                'fpr': np.array([0, 1]),
                'tpr': np.array([0, 1])
            }
        
        # 确保数据类型正确
        pred_ids = pred_ids.astype(np.int64)
        labels = labels.astype(np.int64)
        scores = scores.astype(np.float64)
        
        # 计算基本指标
        precision = precision_score(labels, pred_ids)
        recall = recall_score(labels, pred_ids)
        f1 = f1_score(labels, pred_ids)
        accuracy = accuracy_score(labels, pred_ids)
        
        # 计算ROC相关指标
        fpr, tpr, thresholds = roc_curve(labels, scores)
        auc_score = auc(fpr, tpr)
        
        # 计算EER
        eer, _ = compute_eer(fpr, tpr, thresholds)
        
        # 验证所有指标
        metrics_dict = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'accuracy': float(accuracy),
            'auc': float(auc_score),
            'eer': float(eer),
            'fpr': fpr,  # 添加fpr
            'tpr': tpr   # 添加tpr
        }
        
        # 检查并处理任何NaN值
        for key in metrics_dict:
            if key not in ['fpr', 'tpr'] and np.isnan(metrics_dict[key]):
                print(f"警告: {key}指标为NaN，设置为0")
                metrics_dict[key] = 0.0
        
        return metrics_dict
        
    except Exception as e:
        print(f"计算指标时出错: {str(e)}")
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0,
            'auc': 0.5,
            'eer': 0.5,
            'fpr': np.array([0, 1]),
            'tpr': np.array([0, 1])
        }
def plot_roc_curves(original_metrics, new_metrics):
    """绘制ROC曲线对比图"""
    plt.figure(figsize=(10, 8))
    
    plt.plot(original_metrics['fpr'], original_metrics['tpr'], 'b-',
             label=f'Original Test (AUC = {original_metrics["auc"]:.3f}, EER = {original_metrics["eer"]:.3f})',
             linewidth=2)
    
    plt.plot(new_metrics['fpr'], new_metrics['tpr'], 'r--',
             label=f'New Test (AUC = {new_metrics["auc"]:.3f}, EER = {new_metrics["eer"]:.3f})',
             linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k:', label='Random Guess')
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves Comparison (User {Config.USER_ID}, Window {Config.WINDOW_SIZE})')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(Config.get_figures_path(), 
                            f'roc_comparison_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    plt.savefig(save_path)
    plt.close()

def plot_metrics_comparison(original_metrics, new_metrics):
    """绘制指标对比图"""
    metrics_to_plot = ['precision', 'recall', 'f1', 'accuracy', 'auc', 'eer']
    metrics_names = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC', 'EER']
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(metrics_names))
    width = 0.35
    
    original_values = [original_metrics[m] for m in metrics_to_plot]
    new_values = [new_metrics[m] for m in metrics_to_plot]
    
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
                            f'metrics_comparison_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    plt.savefig(save_path)
    plt.close()

def save_metrics_to_csv(metrics_dict, test_type):
    """保存指标到CSV文件"""
    results_path = os.path.join(Config.get_results_path(), 
                               f'{test_type}_results_user{Config.USER_ID}.csv')
    
    # 只保存需要的指标
    save_dict = {
        'user_id': Config.USER_ID,
        'window_size': Config.WINDOW_SIZE,
        'recall': metrics_dict['recall'],
        'accuracy': metrics_dict['accuracy'],
        'precision': metrics_dict['precision'],
        'f1': metrics_dict['f1'],
        'auc': metrics_dict['auc'],
        'eer': metrics_dict['eer']
    }
    
    df = pd.DataFrame([save_dict])
    
    if os.path.exists(results_path):
        existing_df = pd.read_csv(results_path)
        mask = (existing_df['user_id'] == Config.USER_ID) & \
               (existing_df['window_size'] == Config.WINDOW_SIZE)
        if mask.any():
            existing_df.loc[mask] = df.iloc[0]
        else:
            existing_df = pd.concat([existing_df, df], ignore_index=True)
        df = existing_df
    
    columns = ['user_id', 'window_size', 'recall', 'accuracy', 
              'precision', 'f1', 'auc', 'eer']
    df = df[columns]
    
    df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

def run_testing():
    """主测试函数"""
    try:
        # 加载原始测试数据
        print("\n加载原始测试数据...")
        test_path = os.path.join(Config.get_data_dir(), 'X_test_loader.pkl')
        with open(test_path, 'rb') as f:
            test_dataset = pickle.load(f)
        
        # 检查数据集格式
        inspect_dataset(test_dataset)
        
        # 修改数据加载逻辑
        if isinstance(test_dataset, TensorDataset):
            X_orig = test_dataset.tensors[0]
            y_orig = test_dataset.tensors[1]
        else:
            tensors, labels = [], []
            for data in test_dataset:
                if isinstance(data, (tuple, list)) and len(data) == 2:
                    t, l = data
                    tensors.append(t)
                    labels.append(l)
                
            if not tensors:
                raise ValueError("No valid data found in test dataset")
                
            X_orig = torch.stack(tensors)
            y_orig = torch.stack(labels)
        
        # 确保数据在CPU上并统计样本信息
        X_orig = X_orig.cpu()
        y_orig = y_orig.cpu()
        
        # 统计原始测试集信息
        orig_pos_count = (y_orig == 1).sum().item()
        orig_neg_count = (y_orig == 0).sum().item()
        orig_total = len(y_orig)
        
        print(f"\n原始测试集统计信息:")
        print(f"正样本数量(label=1): {orig_pos_count}")
        print(f"负样本数量(label=0): {orig_neg_count}")
        print(f"总样本数量: {orig_total}")
        print(f"正样本比例: {orig_pos_count/orig_total:.4f}")
        print(f"负样本比例: {orig_neg_count/orig_total:.4f}")
        
        print(f"\nLoaded test data - X shape: {X_orig.shape}, y shape: {y_orig.shape}")
        
        # 加载新测试数据
        print("\n加载新测试数据...")
        num_new_samples = orig_neg_count  # 使用原始负样本数量
        file_path = os.path.join(Config.get_data_dir(), 
                                f'predict_samples_user{Config.USER_ID}_{Config.WINDOW_SIZE}.csv')
        X_new = np.loadtxt(file_path, delimiter=',', skiprows=1)
        X_new = X_new[:num_new_samples]
        X_new = torch.from_numpy(X_new).float().unsqueeze(1)
        y_new = torch.zeros(num_new_samples, dtype=torch.int64)
        
        # 确保新数据在CPU上
        X_new = X_new.cpu()
        y_new = y_new.cpu()
        
        # 统计合并后的数据集信息
        X_combined = torch.cat([X_orig, X_new])
        y_combined = torch.cat([y_orig, y_new])
        
        combined_pos_count = (y_combined == 1).sum().item()
        combined_neg_count = (y_combined == 0).sum().item()
        combined_total = len(y_combined)
        
        print("\n合并后测试集统计信息:")
        print(f"正样本数量(label=1): {combined_pos_count}")
        print(f"负样本数量(label=0): {combined_neg_count}")
        print(f"总样本数量: {combined_total}")
        print(f"正样本比例: {combined_pos_count/combined_total:.4f}")
        print(f"负样本比例: {combined_neg_count/combined_total:.4f}")
        
        print(f"\nLoaded combined data - X shape: {X_combined.shape}, y shape: {y_combined.shape}")
        
        # 创建评估数据加载器
        print("\n创建评估数据加载器...")
        orig_loader = create_evaluation_dataloader(X_orig, y_orig, Config.BATCH_SIZE)
        combined_loader = create_evaluation_dataloader(X_combined, y_combined, Config.BATCH_SIZE)
        
        # 加载模型
        print("\n加载模型...")
        model = MouseNeuralNetwork(X_orig.shape[2]).to(Config.DEVICE)
        model_path = os.path.join(Config.BASE_PATH, 
                                f'pt/gru-only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
        checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=True)
        model.load_state_dict(checkpoint['model'])
        
        # 评估模型
        # 评估模型时添加调试信息
        print("\n评估原始测试集...")
        pred_ids_orig, scores_orig, labels_orig = evaluate_model(model, orig_loader)
        original_metrics = compute_metrics(pred_ids_orig, scores_orig, labels_orig)
        print("\nOriginal metrics computed:")
        print(f"FPR shape: {original_metrics['fpr'].shape}")
        print(f"TPR shape: {original_metrics['tpr'].shape}")
        
        print("\n评估combined测试集...")
        pred_ids_new, scores_new, labels_new = evaluate_model(model, combined_loader)
        new_metrics = compute_metrics(pred_ids_new, scores_new, labels_new)
        print("\nNew metrics computed:")
        print(f"FPR shape: {new_metrics['fpr'].shape}")
        print(f"TPR shape: {new_metrics['tpr'].shape}")
        
        # 绘制可视化结果
        print("\n生成可视化结果...")
        plot_roc_curves(original_metrics, new_metrics)
        plot_metrics_comparison(original_metrics, new_metrics)
        
        # 保存结果
        print("\n保存评估结果...")
        save_metrics_to_csv(original_metrics, 'original_test')
        save_metrics_to_csv(new_metrics, 'new_test')
        
        # 打印评估结果
        print("\n评估结果:")
        print("\nMetric       Original Test    New Test")
        print("-" * 45)
        metrics = ['Precision', 'Recall', 'F1', 'Accuracy', 'AUC', 'EER']
        for metric in metrics:
            metric_key = metric.lower()
            orig_value = original_metrics[metric_key]
            new_value = new_metrics[metric_key]
            print(f"{metric:<12} {orig_value:>8.4f}        {new_value:>8.4f}")
        
        return True
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    run_testing()
