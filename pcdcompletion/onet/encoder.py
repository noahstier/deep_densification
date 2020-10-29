import torch
import torch.nn as nn
from onet.layers import ResnetBlockFC, ResnetBlockConv1dNoBn


def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c

# class ResnetPointnet(nn.Module):
#     ''' PointNet-based encoder network with ResNet blocks.
# 
#     Args:
#         c_dim (int): dimension of latent code c
#         dim (int): input points dimension
#         hidden_dim (int): hidden dimension of the network
#     '''
# 
#     def __init__(self, c_dim=128, dim=3, hidden_dim=128):
#         super().__init__()
#         self.c_dim = c_dim
# 
#         self.fc_pos = nn.Conv1d(dim, 2*hidden_dim, 1)
#         self.block_0 = ResnetBlockConv1dNoBn(2*hidden_dim, size_out=hidden_dim)
#         self.block_1 = ResnetBlockConv1dNoBn(2*hidden_dim, size_out=hidden_dim)
#         self.block_2 = ResnetBlockConv1dNoBn(2*hidden_dim, size_out=hidden_dim)
#         self.block_3 = ResnetBlockConv1dNoBn(2*hidden_dim, size_out=hidden_dim)
#         self.block_4 = ResnetBlockConv1dNoBn(2*hidden_dim, size_out=hidden_dim)
#         self.fc_c = nn.Conv1d(hidden_dim, c_dim, 1)
# 
#         self.actvn = nn.ReLU()
#         self.pool = maxpool
# 
#     def forward(self, p):
#         batch_size, T, D = p.size()
# 
#         p = p.transpose(1, 2)
# 
#         # output size: B x T X F
#         net = self.fc_pos(p)
#         net = self.block_0(net)
#         pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
#         net = torch.cat([net, pooled], dim=1)
# 
#         net = self.block_1(net)
#         pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
#         net = torch.cat([net, pooled], dim=1)
# 
#         net = self.block_2(net)
#         pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
#         net = torch.cat([net, pooled], dim=1)
# 
#         net = self.block_3(net)
#         pooled = self.pool(net, dim=2, keepdim=True).expand(net.size())
#         net = torch.cat([net, pooled], dim=1)
# 
#         net = self.block_4(net)
# 
#         # Recude to  B x F
#         net = self.pool(net, dim=2)
#         net = net.unsqueeze(2)
# 
#         c = self.fc_c(self.actvn(net))
# 
#         c = c.squeeze(-1)
# 
#         return c
