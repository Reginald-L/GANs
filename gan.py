import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets as dataset
from torchvision import transforms as transform
import torch.nn as nn
from torchvision.utils import save_image


# 设置参数
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default="100", help='num_epochs')
parser.add_argument('--img_size', type=int, default=28, help='image size')
parser.add_argument('--channels', type=int, default=1, help='image channels')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='batchsize')
parser.add_argument('--latent_dim', type=int, default=100, help='dimension of noise data')
parser.add_argument('--dataset_path', type=str, default='../../data/mnist', help='dataset path')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu or cpu')
parser.add_argument('--save_path', type=str, default='imgs', help='the path saving imgs')

opt = parser.parse_args()
# 图像形状
img_shape = torch.tensor([opt.channels, opt.img_size, opt.img_size])
# 是否使用GPU
cuda = True if opt.use_gpu and torch.cuda.is_available() else False
print(f'使用GPU: {cuda}')

# 定义模型
# 定义generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            *self.fc_bn_relu_block(opt.latent_dim, 128, batch_norm=False),
            *self.fc_bn_relu_block(128, 256),
            *self.fc_bn_relu_block(256, 512),
            *self.fc_bn_relu_block(512, 1024),
            nn.Linear(1024, torch.prod(img_shape).item()),
            nn.Tanh()
        )  

    def fc_bn_relu_block(self, in_features, out_features, batch_norm=True):
        layers = []
        layers.append(nn.Linear(in_features, out_features))
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.LeakyReLU())
        return layers

    def forward(self, z):
        imgs = self.model(z)
        imgs = imgs.view(z.shape[0], *img_shape)
        return imgs

# 定义Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(torch.prod(img_shape).item(), 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, imgs):
        imgs = imgs.view(imgs.shape[0], -1)
        result = self.model(imgs)
        return result

# 实例化Generator和Discriminator
generator = Generator()
discriminator = Discriminator()

# 损失函数
loss = nn.BCELoss()

if cuda:
    generator = generator.cuda()
    discriminator = discriminator.cuda()
    loss = loss.cuda()

# 优化器
generator_optim = torch.optim.Adam(generator.parameters(), lr=opt.lr)
discriminator_optim = torch.optim.Adam(discriminator.parameters(), lr=opt.lr)

# 构造DataLoader
train_loader = DataLoader(
    dataset=dataset.MNIST(root=opt.dataset_path, 
                        train=True, 
                        transform=transform.Compose([
                            transform.Resize(opt.img_size),
                            transform.ToTensor(),
                            transform.Normalize([0.5], [0.5])
                        ]), 
                        download=True),
    batch_size=opt.batch_size,
    shuffle=True
)

# 创建图像保存文件夹
os.makedirs(opt.save_path, exist_ok=True)

# 训练
for epoch in range(opt.epochs):
    for i, (imgs, _) in enumerate(train_loader):
        # 训练generator
        generator_optim.zero_grad()
        # 生成噪声数据
        z = torch.normal(0, 1, size=(imgs.shape[0], opt.latent_dim))
        z = z.cuda() if cuda else z
        # 让Generator用噪声数据生成图像
        generated_imgs = generator(z) # 让生成器生成图像
        result = discriminator(generated_imgs) # 让鉴别器判断是生成的还是真实的
        # 生成器的训练目的就是让它生成的结果尽最大可能骗过鉴别器, 也就是让鉴别器的结果接近1
        one_result = torch.ones(size=(imgs.shape[0], 1), requires_grad=False)
        one_result = one_result.cuda() if cuda else one_result
        # 优化生成器
        g_loss = loss(result, one_result)
        g_loss.backward()
        generator_optim.step()

        # 训练discriminator
        discriminator_optim.zero_grad()
        # 鉴别器的训练目的包括两个部分: 1)给真实图像一个高一点的结果 2)给生成的图像一个偏低一点的结果
        real_imgs = imgs.cuda() if cuda else imgs
        real_imgs_loss = loss(discriminator(real_imgs), one_result)
        zero_result = torch.zeros(size=(imgs.shape[0], 1), requires_grad=False)
        zero_result = zero_result.cuda() if cuda else zero_result
        fake_imgs_loss = loss(discriminator(generated_imgs.detach()), zero_result)
        d_loss = real_imgs_loss + fake_imgs_loss
        # 优化鉴别器
        d_loss.backward()
        discriminator_optim.step()

        print(f'[epoch: {epoch + 1}/{opt.epochs} batch: {i}/{len(train_loader)} g_loss: {float(g_loss):.6f} d_loss: {float(d_loss):.6f}]')
    
    # 一个epoch结束, 保存一次结果图
    fp = os.path.join(opt.save_path, '%s.png'%(str(epoch + 1)))
    save_image(generated_imgs[:64], fp=fp, nrow=8, normalize=True)