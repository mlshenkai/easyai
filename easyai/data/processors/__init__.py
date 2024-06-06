# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/5/15 5:41 PM
# @File: __init__.py
# @Email: mlshenkai@163.com
from easyai.data.processors.base_processor import BaseProcessor
from easyai.data.processors.alpro_processors import (
    AlproVideoEvalProcessor,
    AlproVideoTrainProcessor,
)
from easyai.data.processors.audio_processors import VideoFileClip
from easyai.data.processors.blip_diffusion_processors import (
    BlipDiffusionInputImageProcessor,
)
from easyai.data.processors.blip_processors import (
    Blip2ImageTrainProcessor,
    BlipCaptionProcessor,
    BlipImageEvalProcessor,
    BlipQuestionProcessor,
)
from easyai.data.processors.clip_processors import ClipImageTrainProcessor, ClipImageEvalProcessor
from easyai.data.processors.gpt_processors import GPTDialogueProcessor, GPTVideoFeatureProcessor
from easyai.common.registry import registry

__all__ = [
    "BaseProcessor",
    "AlproVideoEvalProcessor",
    "AlproVideoTrainProcessor",
    "VideoFileClip",
    "BlipDiffusionInputImageProcessor",
    "BlipCaptionProcessor",
    "BlipImageEvalProcessor",
    "BlipQuestionProcessor",
    "GPTDialogueProcessor",
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
    "GPTVideoFeatureProcessor",
    "GPTDialogueProcessor",
]

def load_processor(name, cfg=None):
    """
    Example

    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor