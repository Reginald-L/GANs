import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import transforms as transforms
from torchvision.utils import save_image


# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='training epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--img_size', type=int, default=28, help='the size of images')
parser.add_argument('--channels', type=int, default=1, help='image channels')
parser.add_argument('--latent_dim', type=int, default=100, help='the dimension of noise data')
parser.add_argument('--n_class', type=int, default=10, help='the number of the classes')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--dataset_path', type=str, default='../../data/fashion_mnist', help='dataset path')
parser.add_argument('--save_path', type=str, default='imgs', help='the path saving imgs')

# 解析参数
opt = parser.parse_args()

# image shape
img_shape = torch.tensor([opt.channels, opt.img_size, opt.img_size])

# 是否使用GPU
cuda = True if opt.use_gpu and torch.cuda.is_available() else False

# 定义模型
class C_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # nn.Embedding(num_embeddings, embedding_dim) :size of the dictionary of embeddings - the size of each embedding vector
        self.embedding = nn.Embedding(opt.n_class, opt.latent_dim)
        
        def block(in_features, out_features, normalise=True):
            layers = []
            layers.append(nn.Linear(in_features, out_features))
            if normalise:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.LeakyReLU())
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim + opt.latent_dim, 128, normalise=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, torch.prod(img_shape)),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 将labels映射到latent_dim的隐空间中
        label_embedding = self.embedding(labels)
        # 将label_embedding和噪声数据拼接在一起
        input = torch.cat((label_embedding, z), -1)
        # 输入到生成器中
        imgs = self.model(input)
        # 将生成的图像形状重置为[B, 1, 28, 28]
        imgs = imgs.view(z.shape[0], *img_shape)
        return imgs


class C_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(opt.n_class, torch.prod(img_shape))

        def block(in_features, out_features, dropout=True):
            layers = []
            layers.append(nn.Linear(in_features, out_features))
            if dropout:
                layers.append(nn.Dropout(0.4, inplace=True))
            layers.append(nn.LeakyReLU())
            return layers

        self.model = nn.Sequential(
            *block(torch.prod(img_shape) + torch.prod(img_shape), 512, dropout=False),
            *block(512, 512),
            *block(512, 512), 
            *block(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, imgs, labels):
        # 将labels映射到latent_dim的隐空间中
        label_embedding = self.embedding(labels)
        # 将imgs从4维转换为2维
        imgs = imgs.view(imgs.shape[0], -1)
        # 将label_embedding和图像拼接在一起
        input = torch.cat((label_embedding, imgs), -1)
        result = self.model(input)
        return result

# 实例化生成器和鉴别器
generator = C_Generator()
discriminator = C_Discriminator()
# 损失函数
loss = nn.BCELoss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss = loss.cuda()

# 优化器
generator_optim = torch.optim.Adam(generator.parameters(), lr=opt.lr)
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

# 数据
train_loader = DataLoader(
    dataset=datasets.FashionMNIST(opt.dataset_path, 
                            train=True, 
                            transform=transforms.Compose([transforms.Resize(opt.img_size), 
                                                          transforms.ToTensor(), 
                                                          transforms.Normalize([0.5], [0.5])]),
                            download=True
                            ),
    batch_size=opt.batch_size,
    shuffle=True
)

# 创建图像保存文件夹
os.makedirs(opt.save_path, exist_ok=True)

# 生成 [0 - 9] 英文 label对应的结果图, 每个label生成10个结果共计 10 * 10 = 100个图
english_lables = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
label_to_index = {label:index for index, label in enumerate(english_lables)}

def sample_image(n_row, fp, cuda=True):
    # Sample noise. Z的shape为 [100, 100]
    z = torch.normal(0, 1, (n_row ** 2, opt.latent_dim))
    z = z.cuda() if cuda else z
    # Get labels ranging from 0 to n_classes for n rows
    labels = torch.tensor([label_to_index[label] for _ in range(n_row) for label in english_lables])
    labels = labels.cuda() if cuda else labels
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data, fp=fp, nrow=n_row, normalize=True)

# 训练
for epoch in range(opt.epochs):
    for i, (imgs, labels) in enumerate(train_loader):
        generator_optim.zero_grad()
        # 生成随机噪声
        z = torch.normal(0, 1, size=(imgs.shape[0], opt.latent_dim))
        z = z.cuda() if cuda else z
        # 用随机噪声和labels生成图像
        real_imgs = imgs.cuda() if cuda else imgs
        labels = labels.cuda() if cuda else labels
        gen_imgs = generator(z, labels)
        # 训练生成器
        # 生成器的训练目的就是让鉴别器尽可能骗过鉴别器获得一个高分
        one_results = torch.ones(size=(imgs.shape[0], 1), requires_grad=False)
        one_results = one_results.cuda() if cuda else one_results
        g_loss = loss(discriminator(gen_imgs, labels), one_results)
        g_loss.backward()
        generator_optim.step()
        # 训练鉴别器
        discriminator_optim.zero_grad()
        real_results = discriminator(real_imgs, labels)
        d_real_loss = loss(real_results, one_results)
        zero_results = torch.zeros(size=(imgs.shape[0], 1), requires_grad=False)
        zero_results = zero_results.cuda() if cuda else zero_results
        # 由于GAN交替训练, 因为训练D时, 我们不希望影响到G. 
        # 只在G生成的fake_img给到D的情况下, 由于不希望更新G的梯度, 所以需要使用detach来截断梯度回流
        # 而在训练生成器的时候, 使用了zero_grad()方法所以不存在这样的问题
        fake_results = discriminator(gen_imgs.detach(), labels)
        d_fake_loss = loss(fake_results, zero_results)
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        discriminator_optim.step()

        print(f'[epoch: {epoch + 1}/{opt.epochs} batch: {i}/{len(train_loader)} g_loss: {float(g_loss):.6f} d_loss: {float(d_loss):.6f}]')
    
    # 一个epoch结束, 保存一次结果图
    fp = os.path.join(opt.save_path, '%s.png'%(str(epoch + 1)))
    sample_image(10, fp, cuda)
