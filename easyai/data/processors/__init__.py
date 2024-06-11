# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:41 PM
# @File: __init__.py
# @Email: mlshenkai@163.com
from easyai.data.processors.base_processor import BaseProcessor

from easyai.data.processors.alpro_processors import (
    AlproVideoTrainProcessor,
    AlproVideoEvalProcessor,
)
from easyai.data.processors.blip_processors import (
    BlipImageTrainProcessor,
    Blip2ImageTrainProcessor,
    BlipImageEvalProcessor,
    BlipCaptionProcessor,
)
from easyai.data.processors.blip_diffusion_processors import (
    BlipDiffusionInputImageProcessor,
    BlipDiffusionTargetImageProcessor,
)
from easyai.data.processors.gpt_processors import (
    GPTVideoFeatureProcessor,
    GPTDialogueProcessor,
)
from easyai.data.processors.clip_processors import ClipImageTrainProcessor
from easyai.data.processors.audio_processors import BeatsAudioProcessor
from easyai.data.processors.ulip_processors import ULIPPCProcessor
from easyai.data.processors.instruction_text_processors import BlipInstructionProcessor

from easyai.common.registry import registry

__all__ = [
    "BaseProcessor",
    # ALPRO
    "AlproVideoTrainProcessor",
    "AlproVideoEvalProcessor",
    # BLIP
    "BlipImageTrainProcessor",
    "Blip2ImageTrainProcessor",
    "BlipImageEvalProcessor",
    "BlipCaptionProcessor",
    "BlipInstructionProcessor",
    # BLIP-Diffusion
    "BlipDiffusionInputImageProcessor",
    "BlipDiffusionTargetImageProcessor",
    # CLIP
    "ClipImageTrainProcessor",
    # GPT
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
    # AUDIO
    "BeatsAudioProcessor",
    # 3D
    "ULIPPCProcessor",
]

def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
