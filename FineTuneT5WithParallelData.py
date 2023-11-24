#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.


####################################################################################################################
#####                                                                                                          #####
#####    This file is an adaptation of the `run_translation.py` script from the huggingface/transformers repo   #####
#####    Reza Alipour                                                                                          #####
#####                                                                                                          #####
####################################################################################################################

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import transformers
from datasets import load_dataset
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    default_data_collator,
    HfArgumentParser
)
from transformers.trainer_utils import get_last_checkpoint, set_seed

logger = logging.getLogger(__name__)
metric = evaluate.load("sacrebleu")


@dataclass
class CustomizedTrainArguments:
    model_name: str = field()
    model_save_name: str = field()
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    model_read_token: str = field(default=None)
    model_write_token: str = field(default=None)
    max_source_length: int = field(default=512)
    max_target_length: int = field(default=512)
    preprocessing_num_workers: int = field(default=1)
    ignore_pad_token_for_loss: bool = field(default=True)
    dataset_name: str = field(default='reza-alipour/Style_Transformer')
    dataset_read_token: str = field(default=None)
    should_log: bool = field(default=True)
    padding: bool = field(default=False)
    training_size: int = field(default=200000)
    training_start_from: int = field(default=0)
    freeze_encoder: bool = field(default=False)


def initialize_logger(should_log, training_args: Seq2SeqTrainingArguments, ):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")


def dataset_preprocess_function(
        examples,
        tokenizer,
        max_source_length,
        max_target_length,
        padding,
        ignore_pad_token_for_loss,
):
    inputs = examples['input']
    targets = examples['output']
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=targets, max_length=max_target_length, padding=padding, truncation=True)
    if ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def compute_metrics(eval_preds, tokenizer):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def main():
    parser = HfArgumentParser((CustomizedTrainArguments, Seq2SeqTrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    initialize_logger(args.should_log, training_args)
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(f'Output directory ({training_args.output_dir}) already exists and is not empty.')
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f'Checkpoint detected, resuming training at {last_checkpoint}.')
    set_seed(training_args.seed)
    dataset = load_dataset(args.dataset_name, token=args.dataset_read_token)
    config = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        revision=args.model_revision,
        token=args.model_read_token,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        revision=args.model_revision,
        token=args.model_read_token,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        pretrained_model_name_or_path=args.model_name,
        revision=args.model_revision,
        config=config,
        token=args.model_read_token,
        trust_remote_code=True
    )

    train_dataset = dataset['train'].select(range(
        args.training_start_from,
        min(args.training_start_from + args.training_size, len(dataset['train']))
    ))
    # eval_dataset = dataset['validation'].select(range(1000))

    preprocessing_lambda = lambda x: dataset_preprocess_function(
        x,
        tokenizer,
        args.max_source_length,
        args.max_target_length,
        padding=args.padding,
        ignore_pad_token_for_loss=args.ignore_pad_token_for_loss
    )
    with training_args.main_process_first(desc='Train dataset map pre-processing'):
        train_dataset = train_dataset.map(
            preprocessing_lambda,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=['input', 'output', 'from', 'type'],
            desc='Running tokenizer on train dataset'
        )
    # with training_args.main_process_first(desc="Validation dataset map pre-processing"):
    #     eval_dataset = eval_dataset.map(
    #         preprocessing_lambda,
    #         batched=True,
    #         num_proc=args.preprocessing_num_workers,
    #         remove_columns=['input', 'output', 'from', 'type'],
    #         desc='Running tokenizer on validation dataset'
    #     )

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if args.padding:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Freezing encoder layers due to low resources
    if args.freeze_encoder:
        for param_name, layer in model.named_parameters():
            if param_name.startswith('encoder'):
                layer.requires_grad = False

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda x: compute_metrics(x, tokenizer)
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    # trainer.save_model()
    tokenizer.push_to_hub(args.model_save_name, private=True, token=args.model_write_token)
    model.push_to_hub(args.model_save_name, private=True, token=args.model_write_token)


if __name__ == "__main__":
    main()
