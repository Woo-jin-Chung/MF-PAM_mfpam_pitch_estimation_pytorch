# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import torch
from torch import nn

class Remix(nn.Module):
    """Remix.
    Mixes different noises with clean speech within a given batch
    """

    def forward(self, sources):
        noise, clean = sources
        bs, *other = noise.shape
        device = noise.device
        perm = torch.argsort(torch.rand(bs, device=device), dim=0)
        return torch.stack([noise[perm], clean])


class Shift(nn.Module):
    """Shift."""

    def __init__(self, shift=8192, same=False):
        """__init__.
        :param shift: randomly shifts the signals up to a given factor
        :param same: shifts both clean and noisy files by the same factor
        """
        super().__init__()
        self.shift = shift
        self.same = same

    def forward(self, wav):
        sources, batch, channels, length = wav.shape
        # 2, B, 1, T  = [2,B,1,T]
        length = length - self.shift
        if self.shift > 0:
            if not self.training:
                wav = wav[..., :length]
            else:
                offsets = torch.randint(
                    self.shift,
                    [1 if self.same else sources, batch, 1, 1], device=wav.device)
                offsets = offsets.expand(sources, -1, channels, -1)
                indexes = torch.arange(length, device=wav.device)
                wav = wav.gather(3, indexes + offsets)
        return wav