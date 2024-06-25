"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
from typing import TYPE_CHECKING

from transformers import (
    PreTrainedModel,
    CONFIG_MAPPING,
    PretrainedConfig,
)

if TYPE_CHECKING:
    from transformers.models.auto.auto_factory import _LazyAutoMapping  # noqa


class Registry:
    mapping = {"model_name_mapping": {}, "model_config_name_mapping": {}}

    @classmethod
    def register_model(cls, name, config_type, model_type: "_LazyAutoMapping"):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.
            config_type:
            model_type:
        Usage:

            from lavis.common.registry import registry
        """

        def wrap(model_cls):
            assert issubclass(
                model_cls, PreTrainedModel
            ), "All models must inherit BaseModel class"
            model_type.register(config_type, model_cls)
            return model_cls

        return wrap

    @classmethod
    def register_model_config(cls, name):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

        def wrap(cfg_cls):
            assert issubclass(
                cfg_cls, PretrainedConfig
            ), "All models must inherit BaseModel class"
            CONFIG_MAPPING.register(f"{cfg_cls.model_type}", cfg_cls)
            return cfg_cls

        return wrap


registry = Registry()
