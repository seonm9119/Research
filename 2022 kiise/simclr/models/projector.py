from torch import nn
from collections import OrderedDict


from simclr.models import get_resnet


class BatchNorm1dNoBias(nn.BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bias.requires_grad = False


class Projector(nn.Module):
    def __init__(self, hparams):
        super().__init__()

        if hparams.arch == 'resnet50':
            self.convnet = get_resnet(hparams=hparams)
            self.encoder_dim = 2048
        elif hparams.arch == 'resnet18':
            self.convnet = get_resnet(hparams=hparams)
            self.encoder_dim = 512
        else:
            raise NotImplementedError

        num_params = sum(p.numel() for p in self.convnet.parameters() if p.requires_grad)

        print(f'======> Encoder: output dim {self.encoder_dim} | {num_params/1e6:.3f}M parameters')

        self.proj_dim = 128

        projection_layers = [
            ('fc1', nn.Linear(self.encoder_dim, self.encoder_dim, bias=False)),
            ('bn1', nn.BatchNorm1d(self.encoder_dim)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(self.encoder_dim, 128, bias=False)),
            ('bn2', BatchNorm1dNoBias(128)),
        ]

        self.projection = nn.Sequential(OrderedDict(projection_layers))

    def forward(self, x, out='z'):
        h = self.convnet(x)
        if out == 'h':
            return h
        return self.projection(h)

