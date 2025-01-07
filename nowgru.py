# gru.py
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import sys
import random
from config import Config
import torch.nn.functional as F
import csv

# Add model path
sys.path.append(Config.get_model_path())
from resnet_mouse import _ResNetLayer
from mouse_traj_classification import MouseNeuralNetwork, MouseNeuralNetwork2
from new_optim import SWATS

def set_random_seed():
    """Set random seeds for reproducibility"""
    torch.manual_seed(Config.RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)

def load_datasets():
    """Load the saved datasets"""
    data_dir = Config.get_data_dir()
    datasets = {}
    
    for name in ['train', 'val', 'test']:
        file_path = os.path.join(data_dir, f'X_{name}_loader.pkl')
        try:
            with open(file_path, 'rb') as f:
                datasets[name] = pickle.load(f)
        except Exception as e:
            print(f"Error loading {name} dataset: {str(e)}")
            raise
    
    return datasets['train'], datasets['val'], datasets['test']

def check_data_batch(data_loader):
    print("\nChecking batch information:")
    for batch_idx, (data, target) in enumerate(data_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target values: {torch.unique(target, return_counts=True)}")
        if batch_idx >= 9:
            break

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """Create DataLoader objects"""
    train_loader = DataLoader(train_dataset, 
                            batch_size=Config.BATCH_SIZE, 
                            shuffle=True,
                            drop_last=True)
    val_loader = DataLoader(val_dataset, 
                          batch_size=64, 
                          shuffle=True,
                          drop_last=True)
    test_loader = DataLoader(test_dataset, 
                           batch_size=Config.BATCH_SIZE, 
                           shuffle=True,
                           drop_last=True)
    
    return train_loader, val_loader, test_loader

    
def train_model(model, train_loader, val_loader, optimizer=None):
    """Train the model"""
    model = model.to(Config.DEVICE)
    writer = SummaryWriter('/home/yanbo.wang/data/CCS2025/Enmouse_case_study/new/tf-logs/')
    
    loss_function = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=Config.LEARNING_RATE,
                                   weight_decay=Config.WEIGHT_DECAY)
    
    # 训练历史记录
    loss_history = []
    val_loss_history = []
    val_acc_history = []
    train_acc_history = []
    epochs = list(range(1, Config.GRU_NUM_EPOCHS + 1))
    
    # 早停相关参数
    best_val_loss = float('inf')
    patience = 10
    min_epochs = 20
    stagnation_counter = 0
    best_model_state = None
    best_epoch = 0
    best_val_acc = 0
    
    for epoch in tqdm(range(Config.GRU_NUM_EPOCHS), desc='Training Progress'):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
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
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        epoch_loss = total_loss / len(train_loader)
        train_acc = correct / total if total > 0 else 0
        loss_history.append(epoch_loss)
        train_acc_history.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE)
                outputs = model(inputs)
                
                val_batch_loss = loss_function(outputs, labels)
                val_loss += val_batch_loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        current_val_loss = val_loss / len(val_loader)
        val_loss_history.append(current_val_loss)
        val_accuracy = val_correct / val_total if val_total > 0 else 0
        val_acc_history.append(val_accuracy)

        # 如果本轮验证集 Loss 比历史最优更低，则更新最佳模型并重置计数器
        if current_val_loss < best_val_loss:
            print(f'Epoch {epoch+1}: Validation loss improved from {best_val_loss:.6f} to {current_val_loss:.6f}')
            best_val_loss = current_val_loss
            best_val_acc = val_accuracy
            stagnation_counter = 0
            best_epoch = epoch
            best_model_state = model.state_dict()

            # 保存最佳模型到本地（可根据需要调整文件名和保存内容）
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': val_accuracy
            }, 'best_model.pth')
        else:
            # 否则计数器+1
            stagnation_counter += 1
            print(f'Epoch {epoch+1}: No improvement in validation loss. '
                  f'Stagnation counter: {stagnation_counter}/{patience}')

        # 只有在达到最小训练轮数后，且连续 patience 轮无改进时，才考虑早停
        if epoch >= min_epochs and stagnation_counter >= patience:
            print(f'\nEarly stopping triggered after {patience} epochs with insufficient improvement')
            print(f'Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}')
            print(f'Best validation accuracy: {best_val_acc:.4f}')

            # 在触发早停后，加载回之前记录的最佳模型权重
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
        
        # Log metrics
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', current_val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)
        
        # Update progress bar description
        tqdm.write(f'Epoch {epoch+1}/{Config.GRU_NUM_EPOCHS} - '
                  f'Loss: {epoch_loss:.4f} - '
                  f'Val Loss: {current_val_loss:.4f} - '
                  f'Train Acc: {train_acc:.4f} - '
                  f'Val Acc: {val_accuracy:.4f} - '
                  f'Stagnation: {stagnation_counter}/{patience}')

    torch.cuda.empty_cache()
    writer.close()

    
    # Plot and save training metrics
    plt.figure(figsize=(12, 8))
    
    # Plot training and validation loss
    plt.subplot(2, 1, 1)
    plt.plot(epochs[:len(loss_history)], loss_history, 'b-', label='Training Loss')
    plt.plot(epochs[:len(val_loss_history)], val_loss_history, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracies
    plt.subplot(2, 1, 2)
    plt.plot(epochs[:len(train_acc_history)], train_acc_history, 'g-', label='Training Accuracy')
    plt.plot(epochs[:len(val_acc_history)], val_acc_history, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    save_path = os.path.join(Config.get_Gruloss_path(), 
                            f'GRU_training_loss_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # Save train and validation losses to CSV
    save_dir = os.path.join(Config.BASE_PATH, 'results', 'GRU_trainvalloss')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create CSV file
    csv_path = os.path.join(save_dir, f'GRU_window{Config.WINDOW_SIZE}_user{Config.USER_ID}_loss.csv')
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])  # Header
        for i in range(len(loss_history)):
            writer.writerow([epochs[i], loss_history[i], val_loss_history[i]])
    
    print(f"Loss values saved to: {csv_path}")
    
    metrics_save_path = os.path.join(Config.get_Gruloss_path(), 
                                   f'training_metrics_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.npz')
    np.savez(metrics_save_path, 
             loss_history=loss_history,
             val_loss_history=val_loss_history,
             train_acc_history=train_acc_history,
             val_acc_history=val_acc_history,
             epochs=epochs[:len(loss_history)])
    
    return model, optimizer, loss_history, val_acc_history, train_acc_history


def run_gru_training():
    """Main GRU training function"""
    try:
        # Set random seed
        set_random_seed()
        
        # Load datasets
        train_dataset, val_dataset, test_dataset = load_datasets()
        
        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        # Get input feature size from sample data
        sample_input, _ = next(iter(train_loader))
        feature_size = sample_input.shape[2]
        
        # Initialize models
        model = MouseNeuralNetwork(feature_size).to(Config.DEVICE)
        model_pretrain = MouseNeuralNetwork2(feature_size).to(Config.DEVICE)
        
        # Load pretrained weights
        pretrained_path = os.path.join(Config.BASE_PATH, 
                                     f'pt/only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
        checkpoint = torch.load(pretrained_path, map_location=Config.DEVICE, weights_only=True)
        model_pretrain.load_state_dict(checkpoint['model'])
        
        # Transfer learning
        model.layer1 = model_pretrain.layer1
#        model.layer2 = model_pretrain.layer2
#        model.layer3 = model_pretrain.layer3
        
        # Initialize optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Train model
        model, optimizer, losses, val_accs, train_accs = train_model(
            model, train_loader, val_loader, optimizer
        )
        
        # Save model
        save_path = os.path.join(Config.BASE_PATH, 
                               f'pt/gru-only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
        torch.save({
            'model': model.state_dict(),
            'epoch': Config.GRU_NUM_EPOCHS
        }, save_path)
        print(f"GRU model saved to: {save_path}")
        
        # Print model parameters
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"GRU model has {params:,} trainable parameters")
        
        return True
        
    except Exception as e:
        print(f"Error in GRU training: {str(e)}")
        raise e

if __name__ == "__main__":
    run_gru_training()