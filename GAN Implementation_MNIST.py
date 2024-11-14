#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
latent_dim = 100
batch_size = 64
num_epochs = 100  
learning_rate = 0.0002
image_dim = 28 * 28  # 784 for MNIST 28x28 images

# Load the MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize the models
generator = Generator(latent_dim, image_dim)
discriminator = Discriminator(image_dim)

# Loss and Optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Fixed noise for generating images at specific epochs
fixed_noise = torch.randn(16, latent_dim)  # 16 images for visualization

# Training
G_losses, D_losses = [], []

def plot_generated_images(epoch, generator, noise):
    with torch.no_grad():
        generated_imgs = generator(noise).view(-1, 1, 28, 28)
        generated_imgs = (generated_imgs + 1) / 2  # Rescale to [0, 1] for visualization
        plt.figure(figsize=(5, 5))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(generated_imgs[i].squeeze(), cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Generated Images at Epoch {epoch}")
        plt.show()

for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        
        # Prepare real and fake labels
        real_imgs = imgs.view(imgs.size(0), -1).to(torch.float32)  # Flatten the image
        real_labels = torch.ones(imgs.size(0), 1)
        fake_labels = torch.zeros(imgs.size(0), 1)
        
        # ============================
        # Train the Discriminator
        # ============================
        optimizer_D.zero_grad()
        
        # Discriminator loss on real images
        outputs = discriminator(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        d_loss_real.backward()
        
        # Generate fake images
        z = torch.randn(imgs.size(0), latent_dim)
        fake_imgs = generator(z)
        
        # Discriminator loss on fake images
        outputs = discriminator(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake.backward()
        
        # Update discriminator
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.step()
        
        # ============================
        # Train the Generator
        # ============================
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

    # Visualize generated images at specific epochs
    if epoch + 1 in [1, 10, 25, 50, 75, 90, 100]:
        plot_generated_images(epoch + 1, generator, fixed_noise)

# Plotting the loss curves
plt.figure(figsize=(10, 5))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.title("Generator and Discriminator Loss During Training")
plt.show()


# In[ ]:




