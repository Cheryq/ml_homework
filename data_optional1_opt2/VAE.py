import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib import pyplot as plt
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 28 * 28, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 32 * 28 * 28)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 32 * 28 * 28)
        self.conv1 = nn.ConvTranspose2d(32, 16, 3)
        self.conv2 = nn.ConvTranspose2d(16, 3, 3)

    def forward(self, z):
        x = F.relu(self.fc1(z))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 32,28, 28)
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x

# 定义VAE模型
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 计算重构损失和KL散度损失

def loss_function(x_recon, x, mu, logvar):
    reconstruction_loss = F.mse_loss(x_recon, x, reduction='sum')
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return reconstruction_loss + 0.5*kl_divergence,reconstruction_loss,kl_divergence

# 训练模型
def train(model, optimizer, train_loader, num_epochs):
    model.train()
    epoch_losses = []
    for epoch in range(num_epochs):
        total_loss = 0
        re_loss=0
        kl=0
        for batch_idx, data in enumerate(train_loader):
            

            optimizer.zero_grad()

            recon_batch, mean, logvar = model(data)
            
            loss,loss1,loss2 = loss_function(recon_batch, data, mean, logvar)
            re_loss+=loss1
            kl+=loss2
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        epoch_losses.append(total_loss)
        print('Epoch {}, Loss: {},re:{},KL:{}'.format(epoch,  total_loss,re_loss,kl))
    plt.plot(range(num_epochs), epoch_losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig('img/loss_curve.png')  # 保存损失曲线图
    torch.save(model.state_dict(), "VAE.pth")

