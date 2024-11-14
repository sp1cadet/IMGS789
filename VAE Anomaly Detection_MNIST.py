#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc_mu(h1)
        logvar = self.fc_logvar(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Load normal test data
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Generate anomalous data (random noise)
anomalous_data = torch.rand((1000, 1, 28, 28))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {train_loss / len(train_loader.dataset):.4f}')

    def compute_reconstruction_error(data_loader):
        model.eval()
        errors = []
        with torch.no_grad():
            for data, _ in data_loader:
                data = data.to(device)
                recon, _, _ = model(data)
                recon_error = torch.mean((recon - data.view(-1, 784)) ** 2, dim=1)
                errors.extend(recon_error.cpu().numpy())
        return np.array(errors)

# Get reconstruction errors for normal and anomalous data
normal_errors = compute_reconstruction_error(test_loader)

# Initialize anomalous_errors as an empty list before the loop
anomalous_errors = []

# Anomalous data evaluation
model.eval()
with torch.no_grad():
    anomalous_errors = []
    for data in anomalous_data:
        data = data.to(device).view(1, 1, 28, 28)
        recon, _, _ = model(data)
        recon_error = torch.mean((recon - data.view(-1, 784)) ** 2, dim=1)
        anomalous_errors.append(recon_error.cpu().item())
        
anomalous_errors = np.array(anomalous_errors)

plt.hist(normal_errors, bins=50, alpha=0.5, label='Normal')
plt.hist(anomalous_errors, bins=50, alpha=0.5, label='Anomalous')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()

threshold = np.percentile(normal_errors, 95)  # 95th percentile of normal data

# Classify samples based on threshold
normal_pred = normal_errors > threshold
anomalous_pred = anomalous_errors > threshold

normal_accuracy = np.mean(~normal_pred)
anomalous_accuracy = np.mean(anomalous_pred)

print(f'Normal accuracy: {normal_accuracy * 100:.2f}%')
print(f'Anomalous accuracy: {anomalous_accuracy * 100:.2f}%')


# In[ ]:




