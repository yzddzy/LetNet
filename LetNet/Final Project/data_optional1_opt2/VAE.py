import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from dataset import get_data
import matplotlib.pyplot as plt

# VAE实现
class VAE(nn.Module):
    def __init__(self, lat_dim):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(32*32*3, 512),
            nn.ReLU(),
            nn.Linear(512, lat_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(lat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 32*32*3),
            nn.Sigmoid()
        )

    def encode(self, x):
        x = x.view(x.size(0), -1)
        latent = self.encoder(x)
        mean, logvar = torch.split(latent, lat_dim, dim=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mean + epsilon * std

    def decode(self, z):
        x_hat = self.decoder(z)
        x_hat = x_hat.view(x_hat.size(0), 3, 32, 32)
        return x_hat

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

# 训练函数实现
def Train(model, dataloader, epochs):
    model.train()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(1, epochs+1):
        running_loss = 0.0
        for images in dataloader:
            images = images
            optimizer.zero_grad()

            recon_images, mean, logvar = model(images)
            loss = nn.functional.mse_loss(
                recon_images, images, reduction='sum') + 0.05 * torch.sum(logvar.exp() - logvar - 1 + 10*mean.pow(2))
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch}/{epochs}], Loss: {epoch_loss:.4f}")
    print("Train complete!")

    # 测试重建图片效果
    # model.eval()
    # with torch.no_grad():
        # image_index = 9  # Index of the image to reconstruct
        # images = X_train[image_index].unsqueeze(0)
        # _, mu0, logvar0 = model(images)
        # r = model.reparameterize(mu0, logvar0)
        # recon_images = model.decode(r)

    # original_image = images[0].cpu().permute(1, 2, 0)
    # reconstructed_image = recon_images[0].cpu().permute(1, 2, 0)

    # plt.figure(figsize=(8, 4))
    # plt.subplot(1, 2, 1)
    # plt.imshow(original_image)
    # plt.title("Original Image")
    # plt.axis('off')

    # plt.subplot(1, 2, 2)
    # plt.imshow(reconstructed_image)
    # plt.title("Reconstructed Image")
    # plt.axis('off')

    # plt.show()
    
# 插值生成图像函数实现
def Generate_images(model, z1, z2, a_):
    model.eval()

    with torch.no_grad():
        images = []
        for a in a_:
            z = (1 - a) * z1 + a * z2
            z = z
            recon_image = model.decode(z.unsqueeze(0))
            images.append(recon_image)

        images = torch.cat(images, dim=0)
        grid = make_grid(images, nrow=len(a_), normalize=True)
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis('off')
        plt.text(0.5, -0.3, 'a = 0, 0.2, 0.4, 0.6, 0.8, 1', transform=plt.gca().transAxes,
                 fontsize=14, color='red', horizontalalignment='center', verticalalignment='top')

        plt.show()


if __name__ == '__main__':
    
    # 预设参数(潜在向量维度，批处理大小，迭代次数，插值列表)
    lat_dim = 128
    bat_size = 32
    epochs = 500
    a_ = [0, 0.2, 0.4, 0.6, 0.8, 1]

    # 处理数据集
    X_train = get_data('dataset')
    X_train = torch.from_numpy(X_train)
    X_train_loader = DataLoader(
        X_train, batch_size=bat_size, shuffle=True)

    # 初始化VAE
    model = VAE(lat_dim)

    # 训练VAE
    Train(model, X_train_loader, epochs)

    # 选取z1，z2潜在向量并可视化其初始图像
    image1 = X_train[0].unsqueeze(0)
    image2 = X_train[1].unsqueeze(0)

    _, mean1, logvar1 = model(image1)
    z1 = model.reparameterize(mean1, logvar1)

    _, mean2, logvar2 = model(image2)
    z2 = model.reparameterize(mean2, logvar2)

    original_image1 = image1[0].cpu().permute(1, 2, 0)
    original_image2 = image2[0].cpu().permute(1, 2, 0)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image1)
    plt.title("Original Image 1")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(original_image2)
    plt.title("Original Image 2")
    plt.axis('off')

    plt.show()

    # 插值生成图像
    Generate_images(model, z1, z2, a_)
