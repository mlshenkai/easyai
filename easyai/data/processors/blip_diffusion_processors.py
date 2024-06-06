# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:40 PM
# @File: __init__.py
# @Email: mlshenkai@163.com

from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

from easyai.common.registry import registry
from easyai.data.processors.base_processor import BaseProcessor
from easyai.data.processors.blip_processors import BlipImageBaseProcessor


@registry.register_processor("blip_diffusion_inp_image_train")
@registry.register_processor("blip_diffusion_inp_image_eval")
class BlipDiffusionInputImageProcessor(BlipImageBaseProcessor):
    def __init__(
        self,
        image_size=224,
        mean=None,
        std=None,
    ):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 224)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        return cls(image_size=image_size, mean=mean, std=std)


@registry.register_processor("blip_diffusion_tgt_image_train")
class BlipDiffusionTargetImageProcessor(BaseProcessor):
    def __init__(
        self,
        image_size=512,
    ):
        super().__init__()

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __call__(self, item):
        return self.transform(item)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 512)

        return cls(image_size=image_size)
