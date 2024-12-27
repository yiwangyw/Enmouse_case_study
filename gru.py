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

def create_dataloaders(train_dataset, val_dataset, test_dataset):
    """Create DataLoader objects"""
    train_loader = DataLoader(train_dataset, 
                            batch_size=Config.BATCH_SIZE, 
                            shuffle=True)
    val_loader = DataLoader(val_dataset, 
                          batch_size=Config.BATCH_SIZE, 
                          shuffle=True)
    test_loader = DataLoader(test_dataset, 
                           batch_size=Config.BATCH_SIZE, 
                           shuffle=True)
    
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, optimizer):
    """Train the GRU model"""
    model = model.to(Config.DEVICE)
    writer = SummaryWriter('/root/tf-logs')
    loss_function = nn.CrossEntropyLoss()
    
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    
    epoch_pbar = tqdm(range(Config.NUM_EPOCHS), desc='Training Progress')
    
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        batch_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        
        for inputs, labels in batch_pbar:
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
        
        epoch_loss = total_loss / len(train_loader)
        train_acc = correct / total
        loss_history.append(epoch_loss)
        
        # Validation phase
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_correct = 0
                val_total = 0
                
                for inputs, labels in val_loader:
                    inputs = inputs.to(Config.DEVICE)
                    labels = labels.to(Config.DEVICE)
                    outputs = model(inputs).squeeze()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                
                val_acc = val_correct / val_total
                val_acc_history.append(val_acc)
                train_acc_history.append(train_acc)
                
                writer.add_scalar('Training loss', epoch_loss, epoch)
                writer.add_scalar('Training accuracy', train_acc, epoch)
                writer.add_scalar('Validation accuracy', val_acc, epoch)
                
                epoch_pbar.set_postfix({
                    'loss': f'{epoch_loss:.4f}',
                    'train_acc': f'{train_acc:.4f}',
                    'val_acc': f'{val_acc:.4f}'
                })
    
    # Plot training results
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'GRU Training Loss (User {Config.USER_ID}, Window {Config.WINDOW_SIZE})')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(Config.get_figures_path(), 
                            f'gru_training_loss_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    plt.savefig(save_path)
    plt.close()
    
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
        
        # Get input feature size from sample data
        sample_input, _ = next(iter(train_loader))
        feature_size = sample_input.shape[2]
        
        # Initialize models
        model = MouseNeuralNetwork(feature_size).to(Config.DEVICE)
        model_pretrain = MouseNeuralNetwork2(feature_size).to(Config.DEVICE)
        
        # Load pretrained weights
        pretrained_path = os.path.join(Config.BASE_PATH, 
                                     f'数据处理代码/only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
        checkpoint = torch.load(pretrained_path, map_location=Config.DEVICE, weights_only=True)
        model_pretrain.load_state_dict(checkpoint['model'])
        
        # Transfer learning
        model.layer1 = model_pretrain.layer1
        model.layer2 = model_pretrain.layer2
        
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
                               f'数据处理代码/gru-only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
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