from typing import Tuple, Optional

import os
import torch
from torch import nn
from models import Darknet

import backbone.base


class DarkNet53(backbone.base.Base):

    def __init__(self, pretrained: bool, models_dir: Optional[str] = None):
        super().__init__(pretrained)
        self._models_dir = models_dir

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        if 'darknet53' not in dir(self):
            self.backbone()

        return self.darknet53

    def backbone(self, config_path) -> Darknet:  # 删除config_path参数
        if 'darknet53' in dir(self) and self.darknet53 is not None:
            raise Exception(
                "Call 'backbone' before 'features' and 'multiscale_features' method")

        if self._models_dir is not None and self._pretrained:
            self.darknet53 = Darknet(config_path)
            self.darknet53.load_state_dict(torch.load(
                os.path.join(self._models_dir, 'darknet53-5d3b4d8f.pth')
            ))
        else:
            self.darknet53 = Darknet(config_path)

        # list(darknet53.children()) consists of following modules
        #   [0] = Conv2d, [1] = BatchNorm2d, [2] = ReLU,
        #   [3] = MaxPool2d, [4] = Sequential(Bottleneck...),
        #   [5] = Sequential(Bottleneck...),
        #   [6] = Sequential(Bottleneck...),
        #   [7] = Sequential(Bottleneck...),
        #   [8] = AvgPool2d, [9] = Linear

        # children = list(self.darknet53.children())
        # self.features = children[:-3]
        # self.num_features_out = 1024

        # self.hidden = children[-3]
        # self.num_hidden_out = 2048

        return self.darknet53
