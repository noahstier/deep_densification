import torch
import torchvision

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def bn_relu_conv(in_channels, out_channels, ksize=3):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(in_channels),
        torch.nn.ReLU(),
        torch.nn.Conv2d(in_channels, out_channels, ksize, padding=1, bias=False),
    )


class FPN(torch.nn.Module):
    def __init__(self, input_height, input_width, n_classes):
        super().__init__()
        self.fpn = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
            "resnet50", pretrained=True
        )
        self.fpn.requires_grad_(False)

        test_img = torch.zeros((1, 3, input_height, input_width))
        fpn_features = self.fpn(test_img)
        dims = [tuple(f.shape[2:]) for f in fpn_features.values()]

        self.scale_heads = torch.nn.ModuleList(
            [
                bn_relu_conv(256, 128),
                torch.nn.Sequential(
                    bn_relu_conv(256, 128),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                ),
                torch.nn.Sequential(
                    bn_relu_conv(256, 128),
                    torch.nn.Upsample(dims[1], mode="bilinear", align_corners=False),
                    bn_relu_conv(128, 128),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                ),
                torch.nn.Sequential(
                    bn_relu_conv(256, 128),
                    torch.nn.Upsample(dims[2], mode="bilinear", align_corners=False),
                    bn_relu_conv(128, 128),
                    torch.nn.Upsample(dims[1], mode="bilinear", align_corners=False),
                    bn_relu_conv(128, 128),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                ),
            ]
        )

        # self.refiner = torch.nn.Sequential(
        #     bn_relu_conv(128, 128), bn_relu_conv(128, 128),
        # )
        self.refiner = torch.nn.Sequential(
            bn_relu_conv(512, 256), bn_relu_conv(256, 128),
        )

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Conv2d(128, n_classes, 1, bias=False),
        #     torch.nn.Upsample(
        #         (input_height, input_width), mode="bilinear", align_corners=False
        #     ),
        # )

    def forward(self, imgs):
        fpn_features = self.fpn(imgs)
        consolidated_feature = self.refiner(
            torch.cat([self.scale_heads[i](fpn_features[i]) for i in range(4)], dim=1)
        )
        # logits = self.classifier(consolidated_feature)
        logits = 0
        return consolidated_feature, logits
