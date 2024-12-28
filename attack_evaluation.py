import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn import metrics
import os
import logging
import matplotlib.pyplot as plt
from attack_config import AttackConfig
from data_utils import ensure_cpu_tensor
from tqdm import tqdm
import time

def compute_kl_divergence(p, q):
    """计算KL散度"""
    p = np.clip(p, 1e-10, 1)
    q = np.clip(q, 1e-10, 1)
    return np.sum(p * np.log(p / q))

def compute_geometric_similarity(traj1, traj2):
    """计算两组轨迹的几何相似度"""
    # 确保两组数据具有相同的样本数
    min_samples = min(len(traj1), len(traj2))
    traj1 = traj1[:min_samples]
    traj2 = traj2[:min_samples]
    
    # 将形状从(n,1,200)转换为(n,200)
    traj1 = traj1.squeeze(1)
    traj2 = traj2.squeeze(1)
    
    # 计算欧氏距离
    distances = np.sqrt(np.sum((traj1 - traj2) ** 2, axis=1))
    return np.mean(distances)

def generate_attack_samples(wc_dcgan, num_samples):
    """生成攻击样本"""
    samples_list = []
    batch_size = 100
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Generating attack samples"):
            current_batch_size = min(batch_size, num_samples - i * batch_size)
            z = torch.randn(current_batch_size, AttackConfig.LATENT_DIM).to(wc_dcgan.device)
            fake = wc_dcgan.generator(z)
            samples_list.append(fake)
    
    samples = torch.cat(samples_list, dim=0)
    return samples[:num_samples]  # 确保exactly返回请求的样本数

def evaluate_attack(target_model, wc_dcgan, real_data, real_labels, num_samples):
    """评估攻击效果"""
    logging.info("\n========== Starting Attack Evaluation ==========")
    
    target_model.eval()
    wc_dcgan.generator.eval()
    
    with torch.no_grad():
        # 生成攻击样本，确保数量与真实样本相同
        num_real_samples = len(real_data)
        fake_samples = generate_attack_samples(wc_dcgan, num_real_samples)  # 修改这里
        logging.info("Attack samples generated successfully")
        
        # 分批处理预测
        batch_size = 100
        num_batches = (len(real_data) + batch_size - 1) // batch_size
        
        # 对真实样本进行预测
        logging.info("Evaluating real samples...")
        real_probs_list = []
        real_preds_list = []
        
        for i in tqdm(range(num_batches), desc="Processing real samples"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(real_data))
            batch_data = real_data[start_idx:end_idx].to(wc_dcgan.device)
            
            outputs = target_model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            real_probs_list.append(probs.cpu())
            real_preds_list.append(preds.cpu())
        
        real_probs = torch.cat(real_probs_list, dim=0)
        real_preds = torch.cat(real_preds_list, dim=0)
        
        # 对攻击样本进行预测
        logging.info("Evaluating attack samples...")
        fake_probs_list = []
        fake_preds_list = []
        num_fake_batches = (len(fake_samples) + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_fake_batches), desc="Processing attack samples"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(fake_samples))
            batch_data = fake_samples[start_idx:end_idx]
            
            outputs = target_model(batch_data)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            fake_probs_list.append(probs.cpu())
            fake_preds_list.append(preds.cpu())
        
        fake_probs = torch.cat(fake_probs_list, dim=0)
        fake_preds = torch.cat(fake_preds_list, dim=0)
        
        logging.info("Computing evaluation metrics...")
        
        # 计算攻击成功率
        attack_success_rate = (fake_preds == 0).float().mean().item()
        
        # 计算KL散度
        real_dist = torch.histc(real_data.cpu().flatten(), bins=100).numpy()
        fake_dist = torch.histc(fake_samples.cpu().flatten(), bins=100).numpy()
        real_dist = real_dist / real_dist.sum()
        fake_dist = fake_dist / fake_dist.sum()
        kl_div = compute_kl_divergence(real_dist, fake_dist)
        
        # 计算几何相似度
        geo_sim = compute_geometric_similarity(
            real_data.cpu().numpy(),
            fake_samples.cpu().numpy()
        )
        
        # 计算分类指标
        precision = precision_score(real_labels.cpu(), real_preds.cpu())
        recall = recall_score(real_labels.cpu(), real_preds.cpu())
        f1 = f1_score(real_labels.cpu(), real_preds.cpu())
        accuracy = accuracy_score(real_labels.cpu(), real_preds.cpu())
        
        # 计算ROC曲线和AUC
        fpr, tpr, _ = metrics.roc_curve(real_labels.cpu(), real_probs[:, 1].cpu())
        auc = metrics.auc(fpr, tpr)
        
        # 保存ROC曲线
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Attack Evaluation')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        save_path = os.path.join(
            AttackConfig.get_attack_figures_path(),
            f'attack_roc_user{AttackConfig.USER_ID}_window{AttackConfig.WINDOW_SIZE}.png'
        )
        plt.savefig(save_path)
        plt.close()
        
        # 保存评估结果
        results = {
            'attack_success_rate': attack_success_rate,
            'kl_divergence': kl_div,
            'geometric_similarity': geo_sim,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'auc': auc,
            'fpr': fpr,
            'tpr': tpr
            # 'type_i_error': type_i_error,
            # 'type_ii_error': type_ii_error
        }
        
        # 记录评估结果
        logging.info("\n========== Attack Evaluation Results ==========")
        for key, value in results.items():
            if not isinstance(value, (np.ndarray, list)):
                logging.info(f"{key}: {value:.4f}")
        
        return results

def visualize_attack_samples(real_samples, fake_samples, save_path):
    """可视化攻击样本"""
    logging.info("Generating attack samples visualization...")
    plt.figure(figsize=(15, 5))
    
    # 绘制真实样本
    for i in range(4):
        plt.subplot(2, 4, i+1)
        plt.plot(real_samples[i].cpu().numpy().squeeze())
        plt.title(f'Real Sample {i+1}')
        plt.grid(True)
    
    # 绘制攻击样本
    for i in range(4):
        plt.subplot(2, 4, i+5)
        plt.plot(fake_samples[i].cpu().numpy().squeeze())
        plt.title(f'Attack Sample {i+1}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    # 用于独立测试
    from attack_main import load_test_data, load_target_model
    from train_attack_model import WCDCGAN
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # 加载数据和模型
    logging.info("Loading test data and target model...")
    test_data, test_labels = load_test_data()
    target_model = load_target_model(
        AttackConfig.WINDOW_SIZE,
        AttackConfig.USER_ID
    )
    
    # 创建和加载攻击模型
    sequence_length = test_data.shape[2]
    wc_dcgan = WCDCGAN(sequence_length, AttackConfig.DEVICE)
    
    checkpoint_path = os.path.join(
        AttackConfig.CHECKPOINT_PATH,
        'wc_dcgan_final.pt'
    )
    
    logging.info(f"Loading attack model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    wc_dcgan.generator.load_state_dict(checkpoint['generator_state_dict'])
    
    # 评估攻击
    results = evaluate_attack(
        target_model,
        wc_dcgan,
        test_data,
        test_labels,
        AttackConfig.NUM_ATTACK_SAMPLES
    )