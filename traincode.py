# traincode.py
import torch 
from torch import nn
from torch.utils import data
from torch.autograd import Variable
import torchvision
from torchvision.datasets import mnist
from torch.utils.data import random_split
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data import SequentialSampler
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import os
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import random
import pickle
from config import Config

# Add model path
sys.path.append(Config.get_model_path())
from resnet_mouse import _ResNetLayer
from mouse_traj_classification import MouseNeuralNetwork, MouseNeuralNetwork2

def set_random_seed():
    """Set random seeds for reproducibility"""
    torch.manual_seed(Config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

def load_data():
    """Load and preprocess data"""
    # Load positive samples (标签为1)
    positive_path = os.path.join(Config.get_data_dir(), 
                               f'positive_samples_user{Config.USER_ID}_{Config.WINDOW_SIZE}.csv')
    data = np.loadtxt(positive_path, delimiter=',', skiprows=1)
    x = torch.from_numpy(data).to(torch.float32).to(Config.DEVICE)
    x = x.unsqueeze(dim=1)
    
    # Load negative samples (标签为0)
    negative_path = os.path.join(Config.get_data_dir(),
                               f'negative_samples_user{Config.USER_ID}_{Config.WINDOW_SIZE}.csv')
    data3 = np.loadtxt(negative_path, delimiter=',', skiprows=1)
    
    # Process negative samples
    data4 = data3[0:]
    np.savetxt('mouse_mov_speeds-user', data4, delimiter=',')
    data5 = np.loadtxt('mouse_mov_speeds-user', delimiter=',')
    y = torch.from_numpy(data5).to(torch.float32).to(Config.DEVICE)
    y = y.unsqueeze(dim=1)
    
    # Create labels: 正样本标签为1，负样本标签为0
    labelx = torch.ones(len(x)).to(torch.int64).to(Config.DEVICE)  # 正样本标签为1
    labely = torch.zeros(len(y)).to(torch.int64).to(Config.DEVICE)  # 负样本标签为0
    label = torch.cat([labelx, labely], dim=0)
    X = torch.cat([x, y], dim=0)
    
    return X, label

def split_and_save_dataset(X, label):
    """Split dataset and save to files with balanced class distribution"""
    # 获取正负样本索引
    pos_idx = (label == 1).nonzero().squeeze()
    neg_idx = (label == 0).nonzero().squeeze()
    
    print(f"Total samples: {len(X)}")
    print(f"Positive samples: {len(pos_idx)}")
    print(f"Negative samples: {len(neg_idx)}")
    
    # 计算划分长度
    pos_train_len = math.floor(len(pos_idx) * Config.TRAIN_SPLIT)
    pos_val_len = math.floor(len(pos_idx) * Config.VAL_SPLIT)
    neg_train_len = math.floor(len(neg_idx) * Config.TRAIN_SPLIT)
    neg_val_len = math.floor(len(neg_idx) * Config.VAL_SPLIT)
    
    # 随机打乱索引
    torch.manual_seed(Config.RANDOM_SEED)  # 确保可重复性
    pos_perm = torch.randperm(len(pos_idx))
    neg_perm = torch.randperm(len(neg_idx))
    
    # 创建划分索引
    train_indices = torch.cat([
        pos_idx[pos_perm[:pos_train_len]],
        neg_idx[neg_perm[:neg_train_len]]
    ])
    val_indices = torch.cat([
        pos_idx[pos_perm[pos_train_len:pos_train_len+pos_val_len]],
        neg_idx[neg_perm[neg_train_len:neg_train_len+neg_val_len]]
    ])
    test_indices = torch.cat([
        pos_idx[pos_perm[pos_train_len+pos_val_len:]],
        neg_idx[neg_perm[neg_train_len+neg_val_len:]]
    ])
    
    # 再次随机打乱
    train_indices = train_indices[torch.randperm(len(train_indices))]
    val_indices = val_indices[torch.randperm(len(val_indices))]
    test_indices = test_indices[torch.randperm(len(test_indices))]
    
    # 创建数据集
    train_dataset = TensorDataset(X[train_indices], label[train_indices])
    val_dataset = TensorDataset(X[val_indices], label[val_indices])
    test_dataset = TensorDataset(X[test_indices], label[test_indices])
    
    # 打印每个集合的样本数量和分布
    print("\nDataset split statistics:")
    for name, dataset in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        total = len(dataset)
        pos = sum(y.item() == 1 for _, y in dataset)
        neg = total - pos
        print(f"{name} set - Total: {total}, Positive: {pos}, Negative: {neg}, Pos ratio: {pos/total:.2%}")
    
    # 保存数据集
    data_dir = Config.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    
    for name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        save_path = os.path.join(data_dir, f'X_{name}_loader.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    print("\nDatasets successfully saved as .pkl files.")
    return train_dataset, val_dataset, test_dataset




# def check_data_batch(train_loader):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         print(f"Batch {batch_idx}:")
#         print(f"Data shape: {data.shape}")
#         print(f"Target shape: {target.shape}")
#         print(f"Target values: {torch.unique(target, return_counts=True)}")
#         if batch_idx == 9:  # ����9������
#             break

# ��train_ADAM������ʼʱ��ӣ�



# ������μ�麯��
def check_data_batch(train_loader):
    print("\nChecking batch information:")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target values: {torch.unique(target, return_counts=True)}")
        if batch_idx >= 9:  # ���ǰ10������
            break


# Add this class for Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.11, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-7

    def forward(self, inputs, targets):
        # 数据验证
        assert inputs.dim() == 2 or inputs.dim() == 1
        assert targets.dim() == 1 or targets.dim() == 2

        # 获取sigmoid后的预测概率
        p = torch.sigmoid(inputs)
        p = torch.clamp(p, self.eps, 1.0 - self.eps)
        
        # 统一维度处理
        targets = targets.float()
        if inputs.size(1) == 2:
            p = p[:, 1].unsqueeze(1)
        if targets.dim() > 1:
            targets = targets.squeeze()
            
        # 计算交叉熵
        ce_loss = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        
        # 计算权重
        weights = torch.where(targets == 1, 
                            self.alpha * torch.pow(1 - p, self.gamma),
                            (1 - self.alpha) * torch.pow(p, self.gamma))
        
        # 添加数值稳定性检查
        if torch.isnan(weights).any() or torch.isinf(weights).any():
            print("Warning: NaN or Inf detected in weights")
            weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
            
        focal_loss = weights * ce_loss
        return focal_loss.mean()


def train_ADAM(model, train_loader, val_loader=None, optimizer=None):
    """Train with Adam optimizer"""
    model = model.to(Config.DEVICE)
    writer = SummaryWriter('/data/yanbo.wang/CCS2025/Enmouse_case_study/new/tf-logs')
    
    # 初始化 Focal Loss
    loss_function = FocalLoss(alpha=0.11, gamma=2.0)
    print(f"Using Focal Loss with alpha=0.25, gamma=2.0")
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=Config.LEARNING_RATE,
                                   weight_decay=Config.WEIGHT_DECAY)
    
    # 初始化早停变量
    best_val_loss = float('inf')
    best_model = None
    patience = 10
    patience_counter = 0
    
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    epochs = list(range(1, Config.Resnet_NUM_EPOCHS + 1))
    
    epoch_pbar = tqdm(range(Config.Resnet_NUM_EPOCHS), desc='Training Progress')
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        
        for inputs, labels in batch_pbar:
            if inputs.size(0) < Config.BATCH_SIZE:
                continue
                
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # 如果输出是 [batch_size, 2]，取第二列的预测概率
            if outputs.size(1) == 2:
                predicted_probs = torch.sigmoid(outputs[:, 1])
            else:
                predicted_probs = torch.sigmoid(outputs.squeeze())
            
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 预测为1的概率大于0.5时判定为正样本
            predicted = (predicted_probs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        if total > 0:
            epoch_loss = total_loss / len(train_loader)
            train_acc = correct / total
            loss_history.append(epoch_loss)
            train_acc_history.append(train_acc)
            
            if val_loader is not None:
                model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        if inputs.size(0) < Config.BATCH_SIZE:
                            continue
                            
                        inputs = inputs.to(Config.DEVICE)
                        labels = labels.to(Config.DEVICE)
                        outputs = model(inputs)
                        
                        # 如果输出是 [batch_size, 2]，取第二列的预测概率
                        if outputs.size(1) == 2:
                            predicted_probs = torch.sigmoid(outputs[:, 1])
                        else:
                            predicted_probs = torch.sigmoid(outputs.squeeze())
                        
                        val_batch_loss = loss_function(outputs, labels)
                        val_loss += val_batch_loss.item()
                        
                        # 预测为1的概率大于0.5时判定为正样本
                        predicted = (predicted_probs > 0.5).float()
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    model.load_state_dict(best_model)
                    break
                
                if val_total > 0:
                    val_accuracy = val_correct / val_total
                    val_acc_history.append(val_accuracy)
                    
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('Accuracy/train', train_acc, epoch)
                    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                    
                    epoch_pbar.set_postfix({
                        'loss': f'{epoch_loss:.4f}',
                        'val_loss': f'{val_loss:.4f}',
                        'train_acc': f'{train_acc:.4f}',
                        'val_acc': f'{val_accuracy:.4f}'
                    })
    
    writer.close()
    
    # Plot and save training metrics
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs[:len(loss_history)], loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(epochs[:len(train_acc_history)], train_acc_history, 'g-', label='Training Accuracy')
    if val_loader is not None:
        plt.plot(epochs[:len(val_acc_history)], val_acc_history, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    save_path = os.path.join(Config.get_Resnetloss_path(), 
                            f'Resnet_training_loss_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    metrics_save_path = os.path.join(Config.get_Resnetloss_path(), 
                                   f'training_metrics_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.npz')
    np.savez(metrics_save_path, 
             loss_history=loss_history,
             train_acc_history=train_acc_history,
             val_acc_history=val_acc_history if val_loader is not None else [],
             epochs=epochs[:len(loss_history)])
    
    return model, optimizer, loss_history, val_acc_history, train_acc_history

    
def run_training():
    """Main training function"""
    try:
        # Set random seed
        set_random_seed()
        
        # Load and preprocess data
        X, label = load_data()
        
        # Split and save datasets
        train_dataset, val_dataset, test_dataset = split_and_save_dataset(X, label)
        
        # Create data loaders with drop_last=True
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=Config.BATCH_SIZE, 
            shuffle=True,
            drop_last=False
        )
        
        # Check batch information before training
        # check_data_batch(train_loader)
        
        # Initialize model
        model = MouseNeuralNetwork2(X.shape[2]).to(Config.DEVICE)
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Train model
        model, optimizer, loss_history, val_acc_history, train_acc_history = train_ADAM(
            model, train_loader, val_loader, optimizer
        )
        
        # Save model
        save_path = os.path.join(Config.BASE_PATH, 
                               f'pt/only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
        torch.save({'model': model.state_dict()}, save_path)
        print(f"Model saved to: {save_path}")
        
        # Print model parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model has {params:,} trainable parameters")
        
        return True
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        raise e

if __name__ == "__main__":
    run_training()