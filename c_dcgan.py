#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :c_dcgan.py
@说明        :条件深度卷积GAN(CDCGAN)
@时间        :2022/12/27 18:59:31
@作者        :Reggie
@版本        :1.0
'''


import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets as datasets
from torchvision import transforms as transformer
from torchvision.utils import save_image

# 设置参数
parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=20, help='training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--n_class', type=int, default=10, help='the number of the classes')
parser.add_argument('--img_size', type=int, default=64, help='image size')
parser.add_argument('--channels', type=int, default=1, help='image channels')
parser.add_argument('--latent_dim', type=int, default=100, help='dimension of noise data')
parser.add_argument('--use_gpu', type=bool, default=True, help='uer gpu')
parser.add_argument('--save_path', type=str, default='imgs', help='path saving result imgs')

opt = parser.parse_args()

cuda = True if opt.use_gpu and torch.cuda.is_available() else False

# 定义模型
# 生成器
class C_DC_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(opt.n_class, opt.latent_dim)

        def block(in_channels, out_channels, k, stride, padding):
            layers = []
            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.init_size = 4
        self.init_channels = 1024

        # 线性层将100维的噪声数据 + 100维的label映射到1024维, 并reshape为1024 * 4 * 4
        self.linear_project = nn.Linear(opt.latent_dim + opt.latent_dim, self.init_channels * self.init_size * self.init_size)

        self.model = nn.Sequential(
            nn.BatchNorm2d(self.init_channels),
            # input: [64, 1024, 4, 4] -> output: [64, 512, 8, 8]
            *block(1024, 512, 4, 2, 1),
            # input: [64, 512, 8, 8] -> output: [64, 256, 16, 16]
            *block(512, 256, 4, 2, 1),
            # input: [64, 256, 16, 16] -> output: [64, 128, 32, 32]
            *block(256, 128, 4, 2, 1),
            # input: [64, 128, 32, 32] -> output: [64, 1, 64, 64]
            nn.ConvTranspose2d(128, opt.channels, 4, 2, 1, bias=False),
            nn.Tanh()            
        )

    def forward(self, z, labels):
        # 将labels映射到latent_dim的隐空间中
        label_embedding = self.embedding(labels)
        # 将label_embedding和噪声数据拼接在一起
        input = torch.cat((label_embedding, z), -1)
        mid_emb = self.linear_project(input)
        mid_emd = mid_emb.view((z.shape[0], self.init_channels, self.init_size, self.init_size))
        imgs = self.model(mid_emd)
        return imgs

# 鉴别器
class C_DC_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.imgs_flatten = nn.Flatten()
        self.embedding = nn.Embedding(opt.n_class, opt.channels * opt.img_size * opt.img_size)

        self.model = nn.Sequential(
            # input: [64, 1, 64, 64] -> output: [64, 64, 32, 32]
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # input: [64, 64, 32, 32] -> output: [64, 128, 16, 16]
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # input: [64, 128, 16, 16] -> output: [64, 256, 8, 8]
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # input: [64, 256, 8, 8] -> output: [64, 512, 4, 4]
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # input: [64, 512, 4, 4] -> output: [64, 1, 1, 1]
            nn.Conv2d(512, 1, 4, 2, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, imgs, labels):
        imgs_flatten = self.imgs_flatten(imgs)
        label_embedding = self.embedding(labels)
        input = (imgs_flatten + label_embedding).view(imgs.shape[0], opt.channels, opt.img_size, opt.img_size)
        result = self.model(input)
        return result.view(imgs.shape[0], 1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# 实例化生成器和鉴别器
generator = C_DC_Generator().apply(weights_init_normal)
discriminator = C_DC_Discriminator().apply(weights_init_normal)
# 实例化损失函数
loss = nn.BCELoss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss = loss.cuda()

# 数据迭代器
train_loader = DataLoader(
    dataset=datasets.MNIST(
        root='../../data/mnist', 
        train=True, 
        transform=transformer.Compose([
            transformer.Resize(opt.img_size),
            transformer.ToTensor(),
            transformer.Normalize([0.5], [0.5])
        ]),
        download=True),
    batch_size=opt.batch_size,
    shuffle=True
)

# 优化器
generator_optim = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# 创建图像保存文件夹
os.makedirs(opt.save_path, exist_ok=True)

# 生成 [0 - 9] label对应的结果图, 每个label生成10个结果共计 10 * 10 = 100个图
def sample_image(n_row, fp, cuda=True):
    # Sample noise. Z的shape为 [100, 100]
    z = torch.normal(0, 1, (n_row ** 2, opt.latent_dim))
    z = z.cuda() if cuda else z
    # Get labels ranging from 0 to n_classes for n rows
    labels = torch.tensor([num for _ in range(n_row) for num in range(n_row)])
    labels = labels.cuda() if cuda else labels
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, fp=fp, nrow=n_row, normalize=True)

for epoch in range(opt.epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        generator_optim.zero_grad()
        # 生成随机噪声数据
        z = torch.normal(0, 1, size=(imgs.shape[0], opt.latent_dim))
        z = z.cuda() if cuda else z
        # 使用噪声数据生成图像
        labels = labels.cuda() if cuda else labels
        gen_imgs = generator(z, labels)
        # 优化生成器
        one_results = torch.ones(size=(imgs.shape[0], 1), requires_grad=False)
        one_results = one_results.cuda() if cuda else one_results
        g_loss = loss(discriminator(gen_imgs, labels), one_results)
        # 反向传播
        g_loss.backward()
        # 更新
        generator_optim.step()

        # 优化鉴别器
        discriminator_optim.zero_grad()
        real_imgs = imgs.cuda() if cuda else imgs
        d_real_loss = loss(discriminator(real_imgs, labels), one_results)
        zero_results = torch.zeros(size=(imgs.shape[0], 1), requires_grad=False)
        zero_results = zero_results.cuda() if cuda else zero_results
        d_fake_loss = loss(discriminator(gen_imgs.detach(), labels), zero_results)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        discriminator_optim.step()

        print(f'[epoch: {epoch + 1}/{opt.epochs} batch: {i}/{len(train_loader)} g_loss: {float(g_loss):.6f} d_loss: {float(d_loss):.6f}]')
    
    # 一个epoch结束, 保存一次结果图
    fp = os.path.join(opt.save_path, '%s.png'%(str(epoch + 1)))
    sample_image(10, fp, cuda)