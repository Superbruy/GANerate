import torch
import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        self.nz = nz
        self.ngf = ngf
        self.nc = nc
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(self.nz, self.ngf * 64, 3, 1, 0),
            nn.BatchNorm2d(self.ngf * 64),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 64, self.ngf * 32, 3, 2, 0),
            nn.BatchNorm2d(self.ngf * 32),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 32, self.ngf * 16, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 16),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 16, self.ngf * 8, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1),
            nn.BatchNorm2d(self.nc),
            nn.Tanh(),
        )

    def forward(self, z):
        output = self.layer(z)
        return output


class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        self.ndf = ndf
        self.nc = nc
        self.layer = nn.Sequential(
            nn.Conv2d(self.nc, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 16, self.ndf * 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 32),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 32, self.ndf * 64, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.ndf * 64),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 64, self.ndf * 128, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer(x)
        return x


def main():
    z = torch.randn(16, 100, 1, 1)
    model = Generator(100, 8, 3)
    print(model(z).shape)

    a = torch.randn(16, 3, 448, 448)
    modelD = Discriminator(8, 3)
    print(modelD(a).shape)

if __name__ == '__main__':
    main()