

import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from dcModel import Discriminator, Generator
import visdom

# set random seed
manual_seed = 1234 # range from 1 to 10000
random.seed(manual_seed)
torch.manual_seed(manual_seed)



# set inputs

data_root = 'C:/Users/CJR/Desktop/documents/data/side'
# number of workers for dataloader
workers = 2
batchsz = 16
learning_rate = 2e-4
image_size = 448
nc = 3
nz = 100
ngf = 16
ndf = 16
ngpu = 1



Transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
datasets = torchvision.datasets.ImageFolder(os.path.join(data_root, 'train'), transform=Transform)
dataloader = DataLoader(datasets, batch_size=batchsz, shuffle=True)

device = torch.device('cuda:0' if (torch.cuda.is_available() and ngpu > 0) else 'cpu')
viz = visdom.Visdom()



# plot some original data
testing_batch, testing_label = next(iter(dataloader))
print(testing_batch.shape)
viz.images(testing_batch, nrow=4, win='show', opts=dict(title='train_images'))
# plt.figure(figsize=(8, 8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(testing_batch.to(device), nrow=8, padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.imshow(np.transpose(vutils.make_grid(testing_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))



# weight initialization
def weight_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0., 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1., 0.02)
        nn.init.constant_(m.bias.data, 0)




# create D and G
G = Generator(nz = nz, ngf=ngf, nc=nc)
D = Discriminator(ndf=ndf, nc=nc)
G.apply(weight_init).to(device)
D.apply(weight_init).to(device)




# loss function
criterion = nn.BCELoss()
init_noise = torch.randn(batchsz, nz, 1, 1, device=device)

optimizer_D = optim.Adam(D.parameters(), lr=learning_rate, betas=(0.5, 0.99))
optimizer_G = optim.Adam(G.parameters(), lr=learning_rate, betas=(0.5, 0.99))

# set labels
real = 1
fake = 0




# train loop

viz.line([[0., 0.]], [0.], win='loss', opts=dict(title='loss', legend=['D', 'G']))
global_step = 0

for epoch in range(1000):
    for batch_idx, (x, _) in enumerate(dataloader):
        x = x.to(device)

        # train Discriminator       train it for k times?
        D.zero_grad()

        # make the real part bigger
        label = torch.full((x.size(0),), real, device=device)
        output_D = D(x).view(-1)
        loss_real = criterion(output_D, label)
        D_x = output_D.mean().item()

        # let the fake part smaller
        fake_label = label.fill_(0.)
        output = G(init_noise).detach()
        get = D(output).view(-1)
        loss_fake = criterion(get, fake_label)
        D_z1 = get.mean().item()

        loss_D = loss_fake + loss_real
        loss_D.backward()
        optimizer_D.step()


        # train Generator
        G.zero_grad()

        out = D(output).view(-1)
        G_label = label.fill_(1.)
        loss = criterion(out, G_label)
        D_z2 = out.mean().item()
        loss.backward()
        optimizer_G.step()

        viz.line([[loss_D.item(), loss.item()]], [global_step], win='loss', update='append')
        print('epoch:{} lossD:{:.4f} lossG:{:.4f} D1:{:.4f} D2:{:.4f} G1:{:.4f} '\
              .format(epoch, loss_D.item(), loss.item(), D_x, D_z1, D_z2))


    # check the quality of the new images
    if epoch % 5 == 0:
        viz.images(G(init_noise), nrow=4, win='generate', opts=dict(title='generate'))



# save model
# torch.save(G.state_dict(), 'ge.mdl')
torch.save(G, 'generator.pth')
