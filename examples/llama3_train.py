# %%
import sys

sys.path.append("/code-online/code/easy_ai")
import os

os.environ["GRADIO_SERVER_PORT"] = "7474"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# %%
import sys

from easyai.configs import get_train_args

sys.argv = [
    "cli",
    "train",
    "--stage",
    "sft",
    "--do_train",
    "True",
    "--model_name_or_path",
    "/code-online/modelscope/llama3-chinese-Instruct",
    "--output_dir",
    "/code/logs",
    "--template",
    "default",
    "--dataset_dir",
    "/code-online/code/easy_ai/data",
    "--dataset",
    "alpaca_zh",
    "--finetuning_type",
    "full",
]
# %%
sys.argv.pop(1)
(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    generating_args,
) = get_train_args()
# %%
from easyai.data import get_dataset
from easyai.models import load_tokenizer, load_model

# %%
tokenizer = load_tokenizer(model_args)
# %%
# datasets = get_dataset(model_args=model_args, data_args=data_args,training_args=training_args,stage="pt", **tokenizer)

# %%
model = load_model(
    tokenizer["tokenizer"],
    model_args,
    finetuning_args,
    is_trainable=True,
    add_valuehead=False,
)
# %%
data_args
# %%
