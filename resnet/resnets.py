import torch
import torch.nn as nn

class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride = 1, downsample=False):
        """
        The main building block of ResNet18 and 34
        Args:
        in_channels: input channels
        out_channels: output channels
        kernel_size: kernel size
        """
        super().__init__()
        self.conv1 =nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels, kernel_size=kernel_size, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = Downsample(in_channels, out_channels) if downsample else None
        
    def forward(self, x):
        identity = x
        
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        
        res = self.conv2(res)
        res = self.bn2(res)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        res += identity
        res = self.relu(res)
        return res

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down= nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=(1,1), stride=(2,2)),
                                 nn.BatchNorm2d(out_channels))
                                 
    
    def forward(self, x):
        return self.down(x)

class ResNet18(nn.Module):
    base_width = 64
    def __init__(self, in_channels = 3, num_classes=10):
        
        """
        PyTorch Implementation of ResNet18 by Hazem Ahmed
        Args:
        in_channels: number of input channels
        num_classes: number of desired predicted classes
        NOTE: image sizes must be 224 x 224
        """
        super().__init__()
        
        base_width = ResNet18.base_width
        self.conv1 = nn.Conv2d(in_channels, base_width, (7,7), stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool= nn.MaxPool2d(kernel_size=3, stride= 2)
        
        self.layer1 = nn.Sequential(IdentityBlock(base_width*1, base_width*1),
                          IdentityBlock(base_width*1, base_width*1))
        self.layer2 = nn.Sequential(IdentityBlock(base_width*1, base_width*2, stride=2, downsample=True),
                          IdentityBlock( base_width*2, base_width*2))
        self.layer3 = nn.Sequential(IdentityBlock(base_width*2, base_width*4, stride=2, downsample=True),
                          IdentityBlock( base_width*4, base_width*4))
        self.layer4 = nn.Sequential(IdentityBlock(base_width*4, base_width*8, stride=2, downsample=True),
                          IdentityBlock( base_width*8, base_width*8))
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_width*8, num_classes)
        
        # using the initialization according to: 
        # K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into rectifiers:
        # Surpassing human-level performance on imagenet classification. In
        # ICCV, 2015.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.max_pool(res)

        res = self.layer1(res)
        res = self.layer2(res)
        res = self.layer3(res)
        res = self.layer4(res)

        res = self.avg_pool(res)
        res = torch.flatten(res, 1)
        res = self.fc(res)

        return res
    
if __name__ == "__main__":
    model = ResNet18(in_channels=3, num_classes=2)

    with torch.no_grad():
        out = model(torch.randn(1,3,224,224))
        print(out)