#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
latent_dim = 100
batch_size = 128
num_epochs = 50  # Reduced to 50 epochs
learning_rate = 0.0002
image_size = 64  # Resize CIFAR-10 to 64x64

# Transform for CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the CIFAR-10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1, 1)

# Initialize the models
generator = Generator(latent_dim)
discriminator = Discriminator()

# Loss and Optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# Fixed noise for consistent final visualization
fixed_noise = torch.randn(16, latent_dim, 1, 1)

# Training
G_losses, D_losses = [], []

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # Prepare real and fake labels
        real_imgs = imgs.to(torch.float32)
        real_labels = torch.ones(imgs.size(0), 1)
        fake_labels = torch.zeros(imgs.size(0), 1)
        
        # Train the Discriminator
        optimizer_D.zero_grad()
        
        # Discriminator loss on real images
        outputs = discriminator(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
        
        # Generate fake images
        z = torch.randn(imgs.size(0), latent_dim, 1, 1)
        fake_imgs = generator(z)
        
        # Discriminator loss on fake images
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        # Update discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.step()
        
        # Train the Generator
        optimizer_G.zero_grad()
        
        # Generator loss
        outputs = discriminator(fake_imgs)
        g_loss = criterion(outputs, real_labels)  # Train G to make D classify fake as real
        g_loss.backward()
        
        # Update generator
        optimizer_G.step()
        
        # Save losses for plotting
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        
    print(f"Epoch [{epoch + 1}/{num_epochs}] - D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Generator and Discriminator Loss During Training")
plt.show()

# Final visualization of generated images
with torch.no_grad():
    generated_imgs = generator(fixed_noise).cpu().detach()
    generated_imgs = (generated_imgs + 1) / 2  # Rescale to [0, 1] for visualization
    plt.figure(figsize=(5, 5))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.transpose(generated_imgs[i], (1, 2, 0)))
        plt.axis('off')
    plt.suptitle("Generated Images after Training")
    plt.show()

# Visualization of real CIFAR-10 images
real_imgs_batch = next(iter(dataloader))[0][:16]
real_imgs_grid = (real_imgs_batch + 1) / 2  # Rescale to [0, 1]
plt.figure(figsize=(5, 5))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(np.transpose(real_imgs_grid[i].numpy(), (1, 2, 0)))
    plt.axis('off')
plt.suptitle("Real CIFAR-10 Images")
plt.show()


# In[ ]:




