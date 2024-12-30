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

#       �� �D  
def check_data_batch(data_loader):
    print("\nChecking batch information:")
    for batch_idx, (data, target) in enumerate(data_loader):
        print(f"\nBatch {batch_idx}:")
        print(f"Data shape: {data.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target values: {torch.unique(target, return_counts=True)}")
        if batch_idx >= 9:  #    ?10      
            break

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """Create DataLoader objects"""
    train_loader = DataLoader(train_dataset, 
                            batch_size=Config.BATCH_SIZE, 
                            shuffle=True,
                            drop_last=True)  #    drop_last
    val_loader = DataLoader(val_dataset, 
                          batch_size=Config.BATCH_SIZE, 
                          shuffle=True,
                          drop_last=True)  #    drop_last
    test_loader = DataLoader(test_dataset, 
                           batch_size=Config.BATCH_SIZE, 
                           shuffle=True,
                           drop_last=True)  #    drop_last
    
    return train_loader, val_loader, test_loader

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

def train_model(model, train_loader, val_loader, optimizer):
    """Train the GRU model"""
    model = model.to(Config.DEVICE)
    writer = SummaryWriter('/data/yanbo.wang/CCS2025/Enmouse_case_study/new/tf-logs')
    
    # ʹ��Focal Loss���CrossEntropyLoss����ʹ��class weights
    loss_function = FocalLoss(gamma=2)
    print(f"Using Focal Loss with gamma=2")    
    
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    epochs = list(range(1, Config.NUM_EPOCHS + 1))
    
    epoch_pbar = tqdm(range(Config.NUM_EPOCHS), desc='Training Progress')    
    for epoch in epoch_pbar:
        # Training phase
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
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        if total > 0:
            epoch_loss = total_loss / len(train_loader)
            train_acc = correct / total
            loss_history.append(epoch_loss)
            train_acc_history.append(train_acc)
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    if inputs.size(0) < Config.BATCH_SIZE:
                        continue
                        
                    inputs = inputs.to(Config.DEVICE)
                    labels = labels.to(Config.DEVICE)
                    outputs = model(inputs).squeeze()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                
                if val_total > 0:
                    val_acc = val_correct / val_total
                    val_acc_history.append(val_acc)
                    
                    writer.add_scalar('Loss/train', epoch_loss, epoch)
                    writer.add_scalar('Accuracy/train', train_acc, epoch)
                    writer.add_scalar('Accuracy/val', val_acc, epoch)
                    
                    epoch_pbar.set_postfix({
                        'loss': f'{epoch_loss:.4f}',
                        'train_acc': f'{train_acc:.4f}',
                        'val_acc': f'{val_acc:.4f}'
                    })
    
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
    save_path = os.path.join(Config.get_Gruloss_path(), 
                            f'gru_training_metrics_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    # Save metrics data
    metrics_save_path = os.path.join(Config.get_Gruloss_path(), 
                                   f'gru_training_metrics_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.npz')
    np.savez(metrics_save_path, 
             loss_history=loss_history,
             train_acc_history=train_acc_history,
             val_acc_history=val_acc_history if val_loader is not None else [],
             epochs=epochs)
    
    writer.close()
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
        
        #       ��  
        # check_data_batch(train_loader)
        
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
        # model.layer2 = model_pretrain.layer2
        
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
            'epoch': Config.NUM_EPOCHS
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