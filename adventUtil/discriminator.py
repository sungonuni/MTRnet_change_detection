import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, in_ch=2, ndf=64):
        super(Discriminator, self).__init__()
        self.Lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv0 = nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(ndf, ndf *2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf *2, ndf *4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf *4, ndf *8, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf *8, 1, kernel_size=4, stride=2, padding=1)
    
    def forward(self,x):
        out = self.conv0(x)
        out = self.Lrelu(out)
        out = self.conv1(out)
        out = self.Lrelu(out)
        out = self.conv2(out)
        out = self.Lrelu(out)
        out = self.conv3(out)
        out = self.Lrelu(out)
        out = self.conv4(out)
        return out



