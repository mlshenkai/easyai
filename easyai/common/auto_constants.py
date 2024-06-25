# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/25 上午11:58
# @File: auto_constants
# @Email: mlshenkai@163.com
from enum import Enum

from transformers.models.auto import *


class AutoModelEnum:
    MODEL = MODEL_MAPPING
    MODEL_FOR_PRETRAINING = MODEL_FOR_PRETRAINING_MAPPING
    MODEL_WITH_LM_HEAD = MODEL_WITH_LM_HEAD_MAPPING
    MODEL_FOR_CAUSAL_LM = MODEL_FOR_CAUSAL_LM_MAPPING
    MODEL_FOR_CAUSAL_IMAGE_MODELING = MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING
    MODEL_FOR_IMAGE_CLASSIFICATION = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION = (
        MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING
    )
    MODEL_FOR_IMAGE_SEGMENTATION = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING
    MODEL_FOR_SEMANTIC_SEGMENTATION = MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING
    MODEL_FOR_INSTANCE_SEGMENTATION = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING
    MODEL_FOR_UNIVERSAL_SEGMENTATION = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING
    MODEL_FOR_VIDEO_CLASSIFICATION = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING
    MODEL_FOR_VISION_2_SEQ = MODEL_FOR_VISION_2_SEQ_MAPPING
    MODEL_FOR_VISUAL_QUESTION_ANSWERING = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING = (
        MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING
    )
    MODEL_FOR_MASKED_LM = MODEL_FOR_MASKED_LM_MAPPING
    MODEL_FOR_IMAGE = MODEL_FOR_IMAGE_MAPPING
    MODEL_FOR_MASKED_IMAGE_MODELING = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING
    MODEL_FOR_OBJECT_DETECTION = MODEL_FOR_OBJECT_DETECTION_MAPPING
    MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING
    MODEL_FOR_DEPTH_ESTIMATION = MODEL_FOR_DEPTH_ESTIMATION_MAPPING
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
    MODEL_FOR_SEQUENCE_CLASSIFICATION = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING
    MODEL_FOR_QUESTION_ANSWERING = MODEL_FOR_QUESTION_ANSWERING_MAPPING
    MODEL_FOR_TABLE_QUESTION_ANSWERING = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING
    MODEL_FOR_TOKEN_CLASSIFICATION = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
    MODEL_FOR_MULTIPLE_CHOICE = MODEL_FOR_MULTIPLE_CHOICE_MAPPING
    MODEL_FOR_NEXT_SENTENCE_PREDICTION = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING
    MODEL_FOR_AUDIO_CLASSIFICATION = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING
    MODEL_FOR_CTC = MODEL_FOR_CTC_MAPPING
    MODEL_FOR_SPEECH_SEQ_2_SEQ = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING
    MODEL_FOR_AUDIO_FRAME_CLASSIFICATION = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING
    MODEL_FOR_AUDIO_XVECTOR = MODEL_FOR_AUDIO_XVECTOR_MAPPING

    MODEL_FOR_TEXT_TO_SPECTROGRAM = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING

    MODEL_FOR_TEXT_TO_WAVEFORM = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING

    MODEL_FOR_BACKBONE = MODEL_FOR_BACKBONE_MAPPING

    MODEL_FOR_MASK_GENERATION = MODEL_FOR_MASK_GENERATION_MAPPING

    MODEL_FOR_KEYPOINT_DETECTION = MODEL_FOR_KEYPOINT_DETECTION_MAPPING

    MODEL_FOR_TEXT_ENCODING = MODEL_FOR_TEXT_ENCODING_MAPPING

    MODEL_FOR_TIME_SERIES_CLASSIFICATION = MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING

    MODEL_FOR_TIME_SERIES_REGRESSION = MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING

    MODEL_FOR_IMAGE_TO_IMAGE = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING
