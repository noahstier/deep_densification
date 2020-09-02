import torch
import torchvision

transform = torchvision.transforms.Compose(
    [
        # torchvision.transforms.Resize(224),
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
        backbone = torchvision.models.detection.backbone_utils.resnet.resnet50(
            pretrained=True, norm_layer=torch.nn.BatchNorm2d
        )

        backbone.requires_grad_(False)
        backbone.layer4.requires_grad_(True)

        return_layers = {"layer1": 0, "layer2": 1, "layer3": 2, "layer4": 3}
        in_channels_stage2 = backbone.inplanes // 8
        in_channels_list = [
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = 256
        self.fpn = torchvision.models.detection.backbone_utils.BackboneWithFPN(
            backbone, return_layers, in_channels_list, out_channels
        )

        test_img = torch.zeros((1, 3, input_height, input_width))
        fpn_features = self.fpn(test_img)
        dims = [tuple(f.shape[2:]) for f in fpn_features.values()]
        self.dims = dims

        self.scale_heads = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    bn_relu_conv(256, 256),
                    torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(256),
                ),
                torch.nn.Sequential(
                    bn_relu_conv(256, 256),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                    torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(256),
                ),
                torch.nn.Sequential(
                    bn_relu_conv(256, 256),
                    torch.nn.Upsample(dims[1], mode="bilinear", align_corners=False),
                    bn_relu_conv(256, 256),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                    torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(256),
                ),
                torch.nn.Sequential(
                    bn_relu_conv(256, 256),
                    torch.nn.Upsample(dims[2], mode="bilinear", align_corners=False),
                    bn_relu_conv(256, 256),
                    torch.nn.Upsample(dims[1], mode="bilinear", align_corners=False),
                    bn_relu_conv(256, 256),
                    torch.nn.Upsample(dims[0], mode="bilinear", align_corners=False),
                    torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
                    torch.nn.BatchNorm2d(256),
                ),
            ]
        )

        self.refiner = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, 3, padding=1, bias=False),
            bn_relu_conv(256, 256),
            bn_relu_conv(256, 256),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
        )

        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Conv2d(256, n_classes, 1, bias=False),
        #     torch.nn.Upsample(
        #         (input_height, input_width), mode="bilinear", align_corners=False
        #     ),
        # )

    def forward(self, imgs):
        fpn_features = self.fpn(imgs)
        stacked_features = torch.stack(
            [self.scale_heads[i](fpn_features[i]) for i in range(4)], dim=0
        )
        summed_features = torch.nn.functional.relu(torch.sum(stacked_features, dim=0))
        combined_features = self.refiner(summed_features)
        # logits = self.classifier(features)
        logits = 0
        return combined_features, logits, fpn_features
