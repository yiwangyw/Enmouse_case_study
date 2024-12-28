import torch
from torch import nn
import torch.optim as optim
import numpy as np
import os
import logging
from torch.utils.data import DataLoader
from attack_config import AttackConfig
from data_utils import GetLoader, ensure_cpu_tensor
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.autograd as autograd
import time
from tqdm import tqdm
import logging
from attack_config import AttackConfig
import numpy as np

class Generator(nn.Module):
    def __init__(self, sequence_length):
        super(Generator, self).__init__()
        self.sequence_length = sequence_length
        self.latent_dim = AttackConfig.LATENT_DIM
        
        self.main = nn.Sequential(
            # 输入是latent_dim维随机噪声
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, sequence_length),  # 生成单通道数据
            nn.Tanh()
        )

    def forward(self, x):
        output = self.main(x)
        # 将输出reshape为(batch_size, 1, sequence_length)，确保只有1个通道
        return output.view(-1, 1, self.sequence_length)

class Discriminator(nn.Module):
    def __init__(self, sequence_length):
        super(Discriminator, self).__init__()
        self.sequence_length = sequence_length
        
        self.main = nn.Sequential(
            nn.Linear(sequence_length * 2, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.view(-1, self.sequence_length * 2)
        return self.main(x)

class WCDCGAN:
    def __init__(self, sequence_length, device):
        self.device = device
        self.generator = Generator(sequence_length).to(device)
        self.discriminator = Discriminator(sequence_length).to(device)
        
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=AttackConfig.ATTACK_LEARNING_RATE,
            betas=(AttackConfig.BETA1, AttackConfig.BETA2)
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=AttackConfig.ATTACK_LEARNING_RATE,
            betas=(AttackConfig.BETA1, AttackConfig.BETA2)
        )

def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1).to(device)
    alpha = alpha.expand(real_samples.size())
    
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    
    fake = Variable(torch.ones(d_interpolates.size()).to(device), requires_grad=False)
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def train_attack_model(wc_dcgan, real_data, real_labels, num_epochs, batch_size):
    logging.info("Starting attack model training...")
    device = AttackConfig.DEVICE
    
    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(real_data, real_labels),
        batch_size=batch_size,
        shuffle=True
    )
    
    # 记录训练指标
    g_losses = []
    d_losses = []
    
    total_steps = len(dataloader)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0
        
        # 创建进度条
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                   leave=True, ncols=100)
        
        for batch_idx, (real_batch, _) in enumerate(pbar):
            batch_size = real_batch.size(0)
            real_batch = real_batch.to(device)
            
            # ---------------------
            #  训练判别器
            # ---------------------
            wc_dcgan.d_optimizer.zero_grad()
            
            # 生成假样本
            z = torch.randn(batch_size, AttackConfig.LATENT_DIM).to(device)
            fake_batch = wc_dcgan.generator(z)
            
            # 计算判别器损失
            real_validity = wc_dcgan.discriminator(real_batch)
            fake_validity = wc_dcgan.discriminator(fake_batch.detach())
            gradient_penalty = compute_gradient_penalty(
                wc_dcgan.discriminator, real_batch, fake_batch.detach(), device
            )
            
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + \
                    AttackConfig.GP_WEIGHT * gradient_penalty
            
            d_loss.backward()
            wc_dcgan.d_optimizer.step()
            
            epoch_d_loss += d_loss.item()
            
            # 每N_CRITIC次迭代训练一次生成器
            if batch_idx % AttackConfig.N_CRITIC == 0:
                # ---------------------
                #  训练生成器
                # ---------------------
                wc_dcgan.g_optimizer.zero_grad()
                
                # 生成新的假样本
                z = torch.randn(batch_size, AttackConfig.LATENT_DIM).to(device)
                gen_traj = wc_dcgan.generator(z)
                fake_validity = wc_dcgan.discriminator(gen_traj)
                
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                wc_dcgan.g_optimizer.step()
                
                epoch_g_loss += g_loss.item()
                
                # 更新进度条信息
                pbar.set_postfix({
                    'D_loss': f'{d_loss.item():.4f}',
                    'G_loss': f'{g_loss.item():.4f}'
                })
        
        # 计算epoch平均损失
        avg_d_loss = epoch_d_loss / len(dataloader)
        avg_g_loss = epoch_g_loss / (len(dataloader) // AttackConfig.N_CRITIC)
        
        # 记录损失
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # 每个epoch结束后输出信息
        elapsed_time = time.time() - start_time
        logging.info(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}, "
            f"Time: {elapsed_time:.2f}s"
        )
        
        # 保存检查点
        if (epoch + 1) % AttackConfig.SAVE_FREQUENCY == 0:
            checkpoint_path = os.path.join(
                AttackConfig.get_checkpoint_path(),
                f'wcdcgan_checkpoint_epoch_{epoch+1}.pt'
            )
            torch.save({
                'epoch': epoch,
                'generator_state_dict': wc_dcgan.generator.state_dict(),
                'discriminator_state_dict': wc_dcgan.discriminator.state_dict(),
                'g_optimizer_state_dict': wc_dcgan.g_optimizer.state_dict(),
                'd_optimizer_state_dict': wc_dcgan.d_optimizer.state_dict(),
                'g_losses': g_losses,
                'd_losses': d_losses,
            }, checkpoint_path)
            logging.info(f"Checkpoint saved: {checkpoint_path}")
    
    logging.info("Training completed!")
    return wc_dcgan

if __name__ == "__main__":
    # 用于独立测试
    from attack_main import load_test_data
    
    test_data, test_labels = load_test_data()
    sequence_length = test_data.shape[2]
    
    wc_dcgan = WCDCGAN(sequence_length, AttackConfig.DEVICE)
    train_attack_model(
        wc_dcgan,
        test_data,
        test_labels,
        AttackConfig.ATTACK_NUM_EPOCHS,
        AttackConfig.ATTACK_BATCH_SIZE
    )