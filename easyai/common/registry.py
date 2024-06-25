"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from transformers import (
    PreTrainedModel,
    CONFIG_MAPPING,
    PretrainedConfig,
    MODEL_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
)

MODEL_TYPE_MAPPING = {
    "model": MODEL_MAPPING,
    "model_for_pretraining": MODEL_FOR_PRETRAINING_MAPPING,
    "model_with_lm_head": MODEL_WITH_LM_HEAD_MAPPING,
    "model_for_causal_lm": MODEL_FOR_CAUSAL_LM_MAPPING,
    "model_for_causal_image": MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    "model_for_image_classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
}


class Registry:
    mapping = {"model_name_mapping": {}, "model_config_name_mapping": {}}

    @classmethod
    def register_model(cls, name, config_type,  model_type):
        r"""Register a task to registry with key 'name'

        Args:
            name: Key with which the task will be registered.

        Usage:

            from lavis.common.registry import registry
        """

        def wrap(model_cls):
            assert issubclass(
                model_cls, PreTrainedModel
            ), "All models must inherit BaseModel class"
            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_name_mapping"][name]
                    )
                )
            cls.mapping["model_name_mapping"][name] = model_cls
            MODEL_TYPE_MAPPING[model_type].register(config_type, model_cls)
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
            if name in cls.mapping["model_config_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["model_config_name_mapping"][name]
                    )
                )
            cls.mapping["model_config_name_mapping"][name] = cfg_cls
            CONFIG_MAPPING.register(f"{cfg_cls.model_type}", cfg_cls)
            return cfg_cls

        return wrap

    @classmethod
    def register(cls, name, obj):
        r"""Register an item to registry with key 'name'

        Args:
            name: Key with which the item will be registered.

        Usage::

            from lavis.common.registry import registry

            registry.register("config", {})
        """
        path = name.split(".")
        current = cls.mapping["state"]

        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[path[-1]] = obj

    @classmethod
    def get_model_class(cls, name):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def list_models(cls):
        return sorted(cls.mapping["model_name_mapping"].keys())

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        r"""Get an item from registry with key 'name'

        Args:
            name (string): Key whose value needs to be retrieved.
            default: If passed and key is not in registry, default value will
                     be returned with a warning. Default: None
            no_warning (bool): If passed as True, warning when key doesn't exist
                               will not be generated. Useful for MMF's
                               internal operations. Default: False
        """
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
            "writer" in cls.mapping["state"]
            and value == default
            and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        r"""Remove an item from registry with key 'name'

        Args:
            name: Key which needs to be removed.
        Usage::

            from mmf.common.registry import registry

            config = registry.unregister("config")
        """
        return cls.mapping["state"].pop(name, None)


registry = Registry()
