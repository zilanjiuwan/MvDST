import torch
import torch.nn as nn
import torch.nn.functional as F


class my_model(nn.Module):
    def __init__(self, dims):
        super(my_model, self).__init__()
        self.layers1 = nn.Linear(dims[0], dims[1])
        self.layers2 = nn.Linear(dims[0], dims[1])
        self.layers_decoder = nn.Linear(dims[1], dims[0])

    def forward(self, x, is_train=True, sigma=0.01):
        out1 = self.layers1(x)
        out2 = self.layers2(x)
        out1 = F.normalize(out1, dim=1, p=2)
        if is_train:
            out2 = F.normalize(out2, dim=1, p=2) + torch.normal(0, torch.ones_like(out2) * sigma).cpu()
        else:
            out2 = F.normalize(out2, dim=1, p=2)

        z = (out1 + out2) / 2
        decoder_out = self.layers_decoder(z)
        return out1, out2, decoder_out
