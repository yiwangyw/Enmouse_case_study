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
    # Load positive samples
    positive_path = os.path.join(Config.get_data_dir(), 
                               f'positive_samples_user{Config.USER_ID}_{Config.WINDOW_SIZE}.csv')
    data = np.loadtxt(positive_path, delimiter=',', skiprows=1)
    x = torch.from_numpy(data).to(torch.float32).to(Config.DEVICE)
    x = x.unsqueeze(dim=1)
    
    # Load negative samples
    negative_path = os.path.join(Config.get_data_dir(),
                               f'negative_samples_user{Config.USER_ID}_{Config.WINDOW_SIZE}.csv')
    data3 = np.loadtxt(negative_path, delimiter=',', skiprows=1)
    
    # Process negative samples
    data4 = data3[0:]
    np.savetxt('mouse_mov_speeds-user', data4, delimiter=',')
    data5 = np.loadtxt('mouse_mov_speeds-user', delimiter=',')
    y = torch.from_numpy(data5).to(torch.float32).to(Config.DEVICE)
    y = y.unsqueeze(dim=1)
    
    # Create labels
    labelx = torch.zeros((len(x))).to(torch.int64).to(Config.DEVICE)
    labely = (torch.zeros((len(y)))+1).to(torch.int64).to(Config.DEVICE)
    label = torch.cat([labelx, labely], dim=0)
    X = torch.cat([x, y], dim=0)
    
    return X, label

def split_and_save_dataset(X, label):
    """Split dataset and save to files"""
    train_len = math.floor(len(X) * Config.TRAIN_SPLIT)
    val_len = math.floor(len(X) * Config.VAL_SPLIT)
    test_len = len(X) - train_len - val_len
    
    dataset = TensorDataset(X, label)
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(0)
    )
    
    # Save datasets
    data_dir = Config.get_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    
    for name, dataset in [('train', train_dataset), ('val', val_dataset), ('test', test_dataset)]:
        save_path = os.path.join(data_dir, f'X_{name}_loader.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
    
    print("Datasets successfully saved as .pkl files.")
    return train_dataset, val_dataset, test_dataset




# def check_data_batch(train_loader):
#     for batch_idx, (data, target) in enumerate(train_loader):
#         print(f"Batch {batch_idx}:")
#         print(f"Data shape: {data.shape}")
#         print(f"Target shape: {target.shape}")
#         print(f"Target values: {torch.unique(target, return_counts=True)}")
#         if batch_idx == 9:  # 检查第9个批次
#             break

# 在train_ADAM函数开始时添加：



# 添加批次检查函数
def check_data_batch(train_loader):
    print("\nChecking batch information:")
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target values: {torch.unique(target, return_counts=True)}")
        if batch_idx >= 9:  # 检查前10个批次
            break


# Add this class for Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha
        if alpha is not None:
            self.alpha = torch.tensor(alpha).to(Config.DEVICE)

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train_ADAM(model, train_loader, val_loader=None, optimizer=None):
    # Initialize Focal Loss without class weights
    loss_function = FocalLoss(gamma=2)
    print(f"Using Focal Loss with gamma=2")
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    writer = SummaryWriter('/data/yanbo.wang/CCS2025/Enmouse_case_study/new/tf-logs')
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    epochs = list(range(1, Config.NUM_EPOCHS + 1))
    
    epoch_pbar = tqdm(range(Config.NUM_EPOCHS), desc='Training Progress')
    
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
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        if total > 0:
            avg_loss = total_loss / len(train_loader)
            train_accuracy = correct / total
            loss_history.append(avg_loss)
            train_acc_history.append(train_accuracy)
            
            if val_loader is not None:
                model.eval()
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        if inputs.size(0) < Config.BATCH_SIZE:
                            continue
                            
                        inputs = inputs.to(Config.DEVICE)
                        labels = labels.to(Config.DEVICE)
                        
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                
                if total > 0:
                    val_accuracy = correct / total
                    val_acc_history.append(val_accuracy)
                    
                    writer.add_scalar('Loss/train', avg_loss, epoch)
                    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
                    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                    
                    epoch_pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'train_acc': f'{train_accuracy:.4f}',
                        'val_acc': f'{val_accuracy:.4f}'
                    })
            else:
                epoch_pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'train_acc': f'{train_accuracy:.4f}'
                })
    
    writer.close()
    
    # Plot and save training metrics
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs, loss_history, 'b-', label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(epochs, train_acc_history, 'g-', label='Training Accuracy')
    if val_loader is not None:
        plt.plot(epochs, val_acc_history, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(Config.get_Resnetloss_path(), 
                            f'Resnet_training_loss_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # Save loss and accuracy data
    metrics_save_path = os.path.join(Config.get_Resnetloss_path(), 
                                   f'training_metrics_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.npz')
    np.savez(metrics_save_path, 
             loss_history=loss_history,
             train_acc_history=train_acc_history,
             val_acc_history=val_acc_history if val_loader is not None else [],
             epochs=epochs)
    
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
            drop_last=True
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