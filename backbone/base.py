from typing import Tuple, Type
from torch import nn


class Base(object):

    OPTIONS = ['darknet53']

    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'darknet53':
            from backbone.darknet53 import DarkNet53
            return DarkNet53
        elif name == 'mobilenetv2':
            raise NotImplementedError
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        raise NotImplementedError

    def backbone(self) -> nn.Module:
        raise NotImplementedError
