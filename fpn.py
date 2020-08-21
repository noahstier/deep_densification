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


def conv_bn_relu(in_channels, out_channels):
    return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
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
                conv_bn_relu(256, 128),
                torch.nn.Sequential(
                    conv_bn_relu(256, 128),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                ),
                torch.nn.Sequential(
                    conv_bn_relu(256, 128),
                    torch.nn.Upsample(dims[1], mode="bilinear", align_corners=False),
                    conv_bn_relu(128, 128),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                ),
                torch.nn.Sequential(
                    conv_bn_relu(256, 128),
                    torch.nn.Upsample(dims[2], mode="bilinear", align_corners=False),
                    conv_bn_relu(128, 128),
                    torch.nn.Upsample(dims[1], mode="bilinear", align_corners=False),
                    conv_bn_relu(128, 128),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                ),
            ]
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(128, n_classes, 1, bias=False),
            torch.nn.Upsample(
                (input_height, input_width), mode="bilinear", align_corners=False
            ),
        )

    def forward(self, imgs):
        fpn_features = self.fpn(imgs)
        consolidated_feature = sum(
            [self.scale_heads[i](fpn_features[i]) for i in range(4)]
        )
        logits = self.classifier(consolidated_feature)
        return consolidated_feature, logits
