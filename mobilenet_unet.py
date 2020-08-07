import torch
import torchvision

imgs = torch.zeros((1, 3, 224, 298))

mobilenet = torchvision.models.mobilenet_v2()
out = mobilenet(imgs)

activations = []
x = imgs
for module in mobilenet.features._modules.values():
    x = module(x)
    activations.append(x)


upsamplers = [
    torch.nn.UpsamplingBilinear2d(size=activations[13].shape[2:]),
    torch.nn.UpsamplingBilinear2d(size=activations[6].shape[2:]),
]

convs = [torch.nn.Conv2d(1376, 1024, 3)]


x = convs[0](torch.cat((activations[13], upsamplers[0](activations[-1])), dim=1))


up2 = upsamplers[1](activations[13])
