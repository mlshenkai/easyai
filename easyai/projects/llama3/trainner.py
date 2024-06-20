# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/6/17 下午7:18
# @File: trainner
# @Email: mlshenkai@163.com
import argparse

from easyai import tasks


def parse_args():
    parser = argparse.ArgumentParser(description="Train Blip2 model")
    parser.add_argument(
        "--cfg-path",
        type=str,
        default="/code-online/code/easy_ai/easyai/projects/llama3/pretrain.yaml",
        help="path to config file",
    )
    parser.add_argument("--options", nargs="+")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    from easyai.common.config import Config
    from easyai.common.registry import registry
    from transformers import AutoModel
    import torch.distributed
    import os
    import fairscale.nn.model_parallel.initialize as fs_init
    from transformers import LlamaModel

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_SIZE"] = "1"

    # model_config = Llama3Config.from_pretrained("/code-online/modelscope/llama3-chinese-Instruct")

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl")

    model_parallel_size = 1
    if not fs_init.model_parallel_is_initialized():
        if model_parallel_size is None:
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        fs_init.initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1234)
    config = Config(parse_args())

    task = tasks.setup_task(config)
    model = task.build_model(config)

    print(config)
