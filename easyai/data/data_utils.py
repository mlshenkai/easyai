
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from datasets import concatenate_datasets, interleave_datasets

from easyai.common.logging import get_logger


if TYPE_CHECKING:
    pass


logger = get_logger(__name__)


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def infer_max_len(
    source_len: int, target_len: int, max_len: int, reserved_label_len: int
) -> Tuple[int, int]:
    max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - min(max_target_len, target_len)
    return max_source_len, max_target_len


def merge_dataset(
    all_datasets: List[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    if len(all_datasets) == 1:
        return all_datasets[0]
    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning(
                "The samples between different datasets will not be mixed in streaming mode."
            )
        return concatenate_datasets(all_datasets)
    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning(
                "We recommend using `mix_strategy=concat` in non-streaming mode."
            )
        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=training_args.seed,
            stopping_strategy="first_exhausted"
            if data_args.mix_strategy.endswith("under")
            else "all_exhausted",
        )
    else:
        raise ValueError("Unknown mixing strategy.")


def split_dataset(
    dataset: Union["Dataset", "IterableDataset"],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Dict[str, "Dataset"]:
    if training_args.do_train:
        if data_args.val_size > 1e-6:  # Split the dataset
            if data_args.streaming:
                dataset = dataset.shuffle(
                    buffer_size=data_args.buffer_size, seed=training_args.seed
                )
                val_set = dataset.take(int(data_args.val_size))
                train_set = dataset.skip(int(data_args.val_size))
                return {"train_dataset": train_set, "eval_dataset": val_set}
            else:
                val_size = (
                    int(data_args.val_size)
                    if data_args.val_size > 1
                    else data_args.val_size
                )
                dataset = dataset.train_test_split(
                    test_size=val_size, seed=training_args.seed
                )
                return {
                    "train_dataset": dataset["train"],
                    "eval_dataset": dataset["test"],
                }
        else:
            if data_args.streaming:
                dataset = dataset.shuffle(
                    buffer_size=data_args.buffer_size, seed=training_args.seed
                )
            return {"train_dataset": dataset}
    else:  # do_eval or do_predict
        return {"eval_dataset": dataset}
