import torch
import torch.nn as nn
from torchvision.models import swin_v2_b

class FPNHead(nn.Module):
    def __init__(self, num_in, num_mid, num_out):
        super().__init__()

        self.block0 = nn.Conv2d(num_in, num_mid, kernel_size=3, padding=1, bias=False)
        self.block1 = nn.Conv2d(num_mid, num_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        x = nn.functional.relu(self.block0(x), inplace=True)
        x = nn.functional.relu(self.block1(x), inplace=True)
        return x


class FPNSwin(nn.Module):

    def __init__(self, norm_layer, output_ch=3, num_filters=64, num_filters_fpn=128, pretrained=True):
        super().__init__()

        # Feature Pyramid Network (FPN) with four feature maps of resolutions
        # 1/4, 1/8, 1/16, 1/32 and `num_filters` filters for all feature maps.

        self.fpn = FPN(num_filters=num_filters_fpn, norm_layer = norm_layer, pretrained=pretrained)

        # The segmentation heads on top of the FPN

        self.head1 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head2 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head3 = FPNHead(num_filters_fpn, num_filters, num_filters)
        self.head4 = FPNHead(num_filters_fpn, num_filters, num_filters)

        self.smooth = nn.Sequential(
            nn.Conv2d(4 * num_filters, num_filters, kernel_size=3, padding=1),
            norm_layer(num_filters),
            nn.ReLU(),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters // 2, kernel_size=3, padding=1),
            norm_layer(num_filters // 2),
            nn.ReLU(),
        )

        self.final = nn.Conv2d(num_filters // 2, output_ch, kernel_size=3, padding=1)

    def unfreeze(self):
        self.fpn.unfreeze()

    def forward(self, x):
        _, map1, map2, map3, map4 = self.fpn(x)

        map4 = nn.functional.upsample(self.head4(map4), scale_factor=8, mode="nearest")
        map3 = nn.functional.upsample(self.head3(map3), scale_factor=4, mode="nearest")
        map2 = nn.functional.upsample(self.head2(map2), scale_factor=2, mode="nearest")

        smoothed = self.smooth(torch.cat([map4, map3, map2, map1], dim=1))
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")
        map0 = nn.functional.upsample(map1, scale_factor=2, mode="bicubic")
        smoothed = self.smooth2(smoothed + map0)
        smoothed = nn.functional.upsample(smoothed, scale_factor=2, mode="nearest")

        final = self.final(smoothed)
        res = torch.tanh(final) + x

        return torch.clamp(res, min = -1, max = 1)


class FPN(nn.Module):

    def __init__(self, norm_layer, num_filters=128, pretrained=False):
        """Creates an `FPN` instance for feature extraction.
        Args:
          num_filters: the number of filters in each output pyramid level
          pretrained: use ImageNet pre-trained backbone feature extractor
        """
 
        super().__init__()
        self.net = swin_v2_b(weights='DEFAULT')

        self.features = self.net.features

        self.enc4 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
        )

        self.enc0 = nn.Sequential(
            self.features[0],
            self.features[1],
        )
        self.enc1 = nn.Sequential(
            self.features[2],
            self.features[3],
        )
        self.enc2 = nn.Sequential(
            self.features[4],
            self.features[5],
        )
        self.enc3 = nn.Sequential(
            self.features[6],
            self.features[7],
        )

        self.td1 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td2 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))
        self.td3 = nn.Sequential(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1),
                                 norm_layer(num_filters),
                                 nn.ReLU(inplace=True))

        self.lateral3 = nn.Conv2d(1024, num_filters, kernel_size=1, bias=False)
        self.lateral2 = nn.Conv2d(512, num_filters, kernel_size=1, bias=False)
        self.lateral1 = nn.Conv2d(256, num_filters, kernel_size=1, bias=False)
        self.lateral0 = nn.Conv2d(128, num_filters // 2, kernel_size=1, bias=False)
        self.lateral4 = nn.Conv2d(32, num_filters // 2, kernel_size=1, bias=False)
        
        for param in self.features.parameters():
            param.requires_grad = False


    def unfreeze(self):
        for param in self.features.parameters():
            param.requires_grad = True


    def forward(self, x):

        # Bottom-up pathway, from ResNet
        enc4 = self.enc4(x)

        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0) # 256
        enc2 = self.enc2(enc1) # 512
        enc3 = self.enc3(enc2) # 1024

        # permute all encoders eg torch.Size([4, 64, 64, 128] to torch.Size([4, 128, 64, 64])
        enc0 = enc0.permute(0, 3, 1, 2)
        enc1 = enc1.permute(0, 3, 1, 2)
        enc2 = enc2.permute(0, 3, 1, 2)
        enc3 = enc3.permute(0, 3, 1, 2)

        # Lateral connections

        lateral4 = self.lateral4(enc4)
        lateral3 = self.lateral3(enc3)
        lateral2 = self.lateral2(enc2)
        lateral1 = self.lateral1(enc1)
        lateral0 = self.lateral0(enc0)
        
        # Top-down pathway
        map3 = lateral3 #self.td1(lateral3 + nn.functional.upsample(map4, scale_factor=2, mode="nearest"))
        map2 = self.td1(lateral2 + nn.functional.upsample(map3, scale_factor=2, mode="nearest"))
        map1 = self.td2(lateral1 + nn.functional.upsample(map2, scale_factor=2, mode="nearest"))
        return lateral4, lateral0, map1, map2, map3#, map4

