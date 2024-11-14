#!/usr/bin/env python
# coding: utf-8

# In[3]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
latent_dim = 20
batch_size = 128
num_epochs = 100
learning_rate = 1e-3

# Load MNIST dataset with normalization to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts images to tensors in the range [0, 1]
])

dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # Encoder
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_logvar = nn.Linear(400, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)

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
        mu, logvar = self.encode(x.view(-1, 28 * 28))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def vae_loss(recon_x, x, mu, logvar):
    # Reconstruction loss
    BCE = nn.functional.binary_cross_entropy(recon_x, x.view(-1, 28 * 28), reduction='sum')
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD, BCE, KLD

# Model, optimizer, and loss variables
vae = VAE(latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

# Lists to store losses
total_loss_list, recon_loss_list, kl_div_list = [], [], []

# Training loop
for epoch in range(num_epochs):
    total_loss, recon_loss, kl_div = 0, 0, 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.view(-1, 28 * 28)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = vae(data)
        loss, recon, kl = vae_loss(recon_batch, data, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        recon_loss += recon.item()
        kl_div += kl.item()

    # Average loss for each epoch
    total_loss_list.append(total_loss / len(dataloader.dataset))
    recon_loss_list.append(recon_loss / len(dataloader.dataset))
    kl_div_list.append(kl_div / len(dataloader.dataset))
    
    print(f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss_list[-1]:.4f}, Reconstruction Loss: {recon_loss_list[-1]:.4f}, KL Divergence: {kl_div_list[-1]:.4f}")

# Plot ELBO loss and KL divergence
plt.figure(figsize=(10, 4))
plt.plot(total_loss_list, label="Total Loss (ELBO)")
plt.plot(recon_loss_list, label="Reconstruction Loss")
plt.plot(kl_div_list, label="KL Divergence")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("ELBO Loss and KL Divergence During Training")
plt.show()

# Visualization of reconstruction
vae.eval()
with torch.no_grad():
    data, _ = next(iter(dataloader))
    data = data.view(-1, 28 * 28)
    recon_batch, _, _ = vae(data)
    recon_batch = recon_batch.view(-1, 1, 28, 28)

    # Display original images
    plt.figure(figsize=(9, 3))
    for i in range(8):
        plt.subplot(2, 8, i + 1)
        plt.imshow(data[i].view(28, 28).cpu().numpy(), cmap='gray')
        plt.axis('off')
    # Display reconstructed images
    for i in range(8):
        plt.subplot(2, 8, i + 9)
        plt.imshow(recon_batch[i].view(28, 28).cpu().numpy(), cmap='gray')
        plt.axis('off')
    plt.suptitle("Original and Reconstructed Images")
    plt.show()

# Generate new images by sampling from the learned latent space
with torch.no_grad():
    z = torch.randn(16, latent_dim)
    generated_images = vae.decode(z).view(-1, 1, 28, 28)

    plt.figure(figsize=(5, 5))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i].cpu().numpy().squeeze(), cmap='gray')
        plt.axis('off')
    plt.suptitle("Generated Images from Latent Space")
    plt.show()


# In[5]:


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


# In[6]:


pip install seaborn


# In[7]:


# Initialize lists to store latent vectors and labels
latent_vectors = []
labels = []

model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        mu, _ = model.encode(data.view(-1, 784))
        latent_vectors.extend(mu.cpu().numpy())
        labels.extend(target.numpy())

# Convert to numpy arrays
latent_vectors = np.array(latent_vectors)
labels = np.array(labels)

import matplotlib.pyplot as plt
import seaborn as sns

# Create a scatter plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_vectors[:, 0], latent_vectors[:, 1], c=labels, cmap='tab10', alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label='Digit Class')
plt.xlabel('Latent Dimension 1')
plt.ylabel('Latent Dimension 2')
plt.title('Latent Space of VAE on MNIST')
plt.show()


# In[ ]:




