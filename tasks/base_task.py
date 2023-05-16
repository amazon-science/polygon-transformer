# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import logging
import os
import math
import torch
from typing import Dict, Optional

from fairseq import search
from fairseq.data import FairseqDataset, iterators, Dictionary
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from omegaconf import DictConfig
from torch import Tensor, device, dtype, nn



logger = logging.getLogger(__name__)


def load_bert_pretrained_weights(model, ckpt_path):
    try:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        raise OSError(
            "Unable to load weights from pytorch checkpoint file. "
            "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
        )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []


    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    ##############################################################################################

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = "bert."
    load(model, prefix=start_prefix)

    if len(unexpected_keys) > 0:
        logger.warning(
            f"Some weights of the model checkpoint at {ckpt_path} were not used when "
            f"initializing {model.__class__.__name__}: {unexpected_keys}\n"
            f"- This IS expected if you are initializing {model.__class__.__name__} from the checkpoint of a model trained on another task "
            f"or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPretraining model).\n"
            f"- This IS NOT expected if you are initializing {model.__class__.__name__} from the checkpoint of a model that you expect "
            f"to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
        )
    else:
        logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
    if len(missing_keys) > 0:
        logger.warning(
            f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {ckpt_path} "
            f"and are newly initialized: {missing_keys}\n"
            f"You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference."
        )
    else:
        logger.info(
            f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at {ckpt_path}.\n"
            f"If your task is similar to the task the model of the ckeckpoint was trained on, "
            f"you can already use {model.__class__.__name__} for predictions without further training."
        )
    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(error_msgs)
            )
        )




@dataclass
class BaseConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "comma separated path to data list, will be iterated upon during epochs "
                    "in round-robin manner; valid data are always in the last"
        },
    )
    selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "selected cols"},
    )
    bpe_dir: Optional[str] = field(
        default=None,
        metadata={"help": "bpe dir"},
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    max_src_length: int = field(
        default=128, metadata={"help": "the maximum src sequence length"}
    )
    max_tgt_length: int = field(
        default=30, metadata={"help": "the maximum target sequence length"}
    )

    code_dict_size: int = field(
        default=8192, metadata={"help": "code dict size"}
    )
    patch_image_size: int = field(
        default=480, metadata={"help": "patch image size"}
    )
    num_bins: int = field(
        default=1000, metadata={"help": "number of quantization bins"}
    )

    imagenet_default_mean_and_std: bool = field(
        default=False,
        metadata={"help": "imagenet normalize"},
    )
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )


@register_task("base_task", dataclass=BaseConfig)
class BaseTask(FairseqTask):
    def __init__(self, cfg: BaseConfig, src_dict, tgt_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""

        # Define dictionaries
        src_dict = Dictionary()
        tgt_dict = Dictionary()

        # Add 2D bin tokens
        for i in range(cfg.num_bins):
            for j in range(cfg.num_bins):
                src_dict.add_symbol("<bin_{}_{}>".format(i, j))
                tgt_dict.add_symbol("<bin_{}_{}>".format(i, j))

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # create mini-batches with given size constraints
        batch_sampler = [
            [j for j in range(i, min(i + max_sentences, len(dataset)))]
            for i in range(0, len(dataset), max_sentences)
        ]
        total_row_count = dataset.dataset.get_total_row_count()
        num_batches = math.ceil(math.ceil(total_row_count / num_shards) / max_sentences)
        if len(batch_sampler) < num_batches:
            batch_sampler.append([])

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=1,
            shard_id=0,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size
        )

        return epoch_iter

    def build_model(self, cfg: FairseqDataclass):
        model = super().build_model(cfg)
        bpe_dict = {
            "_name": "gpt2",
            "gpt2_encoder_json": os.path.join(self.cfg.bpe_dir, "encoder.json"),
            "gpt2_vocab_bpe": os.path.join(self.cfg.bpe_dir, "vocab.bpe")
        }
        bpe_dict = DictConfig(bpe_dict)
        self.bpe = self.build_bpe(bpe_dict)
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False, **extra_kwargs
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = criterion(model, sample, update_num=update_num)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict
