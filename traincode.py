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
from new_optim import SWATS

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

def train_ADAM(model, train_loader, val_loader=None, optimizer=None):
    """Train the model using ADAM optimizer"""
    writer = SummaryWriter('/root/tf-logs')
    loss_function = nn.CrossEntropyLoss()
    
    model = model.to(Config.DEVICE)
    loss_history = []
    val_acc_history = []
    train_acc_history = []
    
    progress_bar = tqdm(range(Config.NUM_EPOCHS), desc='Training', unit='epoch')
    
    for epoch in progress_bar:
        running_loss = 0.0
        batch_count = 0
        correct = 0
        total = 0
        
        model.train()
        batch_progress = tqdm(train_loader, 
                            desc=f'Epoch {epoch+1}/{Config.NUM_EPOCHS}', 
                            leave=False)
        
        for inputs, labels in batch_progress:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = loss_function(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            batch_count += 1
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            batch_progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{correct/total:.4f}'
            })
        
        epoch_loss = running_loss / batch_count
        epoch_acc = correct / total
        loss_history.append(epoch_loss)
        
        progress_bar.set_postfix({
            'loss': f'{epoch_loss:.4f}',
            'acc': f'{epoch_acc:.4f}'
        })
        
        # Validation every 10 epochs
        if epoch % 10 == 0 and val_loader is not None:
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
                train_acc_history.append(epoch_acc)
                
                writer.add_scalar('Training loss', epoch_loss, epoch)
                writer.add_scalar('Training accuracy', epoch_acc, epoch)
                writer.add_scalar('Validation accuracy', val_acc, epoch)
    
    # Plot and save loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, 'b-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Resnet Training Loss (User {Config.USER_ID}, Window {Config.WINDOW_SIZE})')
    plt.grid(True)
    
    save_path = os.path.join(Config.get_figures_path(), 
                            f'Resnet training_loss_user{Config.USER_ID}_window{Config.WINDOW_SIZE}.png')
    plt.savefig(save_path)
    plt.close()
    
    writer.close()
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
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=Config.BATCH_SIZE, 
                                shuffle=True)
        val_loader = DataLoader(val_dataset, 
                              batch_size=Config.BATCH_SIZE, 
                              shuffle=True)
        
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
                               f'数据处理代码/only-adam-user{Config.USER_ID}_{Config.WINDOW_SIZE}-path.pt')
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