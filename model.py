import torch
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1, stride=1),
        )

    def forward(self, image):
        """
            Predict whether the given image is fake (0) or not (1).
            image: (batch, 64, 84, 1)
            return : (batch, 16, 21, 1)
        """
        image_ = image.view(image.shape[0], image.shape[3], image.shape[1], image.shape[2])
        h = self.conv(image_)
        pred = h.view(h.shape[0], h.shape[2], h.shape[3], h.shape[1])
        return pred


class Residual(nn.Module):

    def __init__(self):
        super(Residual, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
        )

    def forward(self, x):
        """
            x: (batch, 256, 16, 21)
            return : (batch, 256, 16, 21)
        """
        y = F.pad(x, (1, 1, 1, 1), "constant", 0)
        y = self.conv(y)
        y = F.pad(y, (1, 1, 1, 1), "constant", 0)
        y = self.conv(y)
        return F.relu(y + x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            Residual(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, stride=1),
            nn.Sigmoid(),
        )

    def forward(self, image):
        """
            Output a fake image with the same size.
            image: (batch, 64, 84, 1)
            return : (batch, 64, 84, 1)
        """
        image_ = image.view(image.shape[0], image.shape[3], image.shape[1], image.shape[2])
        h0 = F.pad(image_, (3, 3, 3, 3), "constant", 0)
        h1 = self.conv1(h0)
        h2 = F.pad(h1, (3, 3, 3, 3), "constant", 0)
        h3 = self.conv2(h2)
        pred = h3.view(h3.shape[0], h3.shape[2], h3.shape[3], h3.shape[1])
        return pred


if __name__ == '__main__':
    import utils.pytorch_util as ptu
    ptu.set_gpu_mode(True)
    d = Discriminator()
    g = Generator()
    r = Residual()
    image = ptu.ones((3, 64, 84, 1))
    # print(d(image).shape)
    image_ = ptu.ones((3, 1, 64, 84))
    x = ptu.ones((3, 256, 16, 21))
    a = F.pad(image_, (3, 3, 3, 3), "constant", 0)
    print(g(image).shape)
    print(d(image).shape)
