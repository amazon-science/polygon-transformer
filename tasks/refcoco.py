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
from typing import Optional
import os
import math
import numpy as np
import torch
from fairseq import metrics
from fairseq.tasks import register_task

from tasks.base_task import BaseTask, BaseConfig, load_bert_pretrained_weights
from data.refcoco_dataset import RefcocoDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


COO = 0  # <COO> class
SEP = 1  # <SEP> class
EOS = 2  # <EOS> class
bos_index = 0   # index for bos token
sep_index = 3  # index for separator token




@dataclass
class RefcocoConfig(BaseConfig):
    eval_acc: bool = field(
        default=False, metadata={"help": "evaluation with accuracy"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    max_image_size: int = field(
        default=512, metadata={"help": "max image size for normalization"}
    )
    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )


@register_task("refcoco", dataclass=RefcocoConfig)
class RefcocoTask(BaseTask):
    def __init__(self, cfg: RefcocoConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = RefcocoDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            num_bins=self.cfg.num_bins,
            max_image_size=self.cfg.max_image_size
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        bert_path = "../../pretrained_weights/bert-base-uncased-pytorch_model.bin"
        if os.path.exists(bert_path):
            load_bert_pretrained_weights(model.encoder.bert, bert_path)
        if cfg._name == 'polyformer_b':
            swin_path = "../../pretrained_weights/swin_base_patch4_window12_384_22k.pth"
        else:
            swin_path = "../../pretrained_weights/swin_large_patch4_window12_384_22k.pth"
        if os.path.exists(swin_path):
            model.encoder.embed_images.init_weights(pretrained=swin_path)
        return model

    def _calculate_ap_score(self, hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)
        model.eval()
        if self.cfg.eval_acc:
            hyps, refs = self._inference(sample, model)
            scores = self._calculate_ap_score(hyps.float(), refs.float())
            logging_output["_score_sum"] = scores.sum().item()
            logging_output["_score_cnt"] = scores.size(0)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)

    def _inference(self, sample, model):
        hyps = self.inference_step(model, sample)
        refs = sample['region_coords'].float()
        hyps = hyps * self.cfg.max_image_size
        hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
        hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)
        return hyps, refs

    def inference_step(self, model, sample):
        with torch.no_grad():
            if isinstance(model, list):
                model = model[0]
            min_len = 6
            max_len = 210
            model.eval()
            img = sample["net_input"]["patch_images"]
            b = img.shape[0]
            prev_output_token_11 = [[bos_index] for _ in range(b)]
            prev_output_token_12 = [[bos_index] for _ in range(b)]
            prev_output_token_21 = [[bos_index] for _ in range(b)]
            prev_output_token_22 = [[bos_index] for _ in range(b)]
            delta_x1 = [[0] for _ in range(b)]
            delta_y1 = [[0] for _ in range(b)]
            delta_x2 = [[1] for _ in range(b)]
            delta_y2 = [[1] for _ in range(b)]

            gen_out = [[] for _ in range(b)]

            n_bins = self.cfg.num_bins

            unfinish_flag = np.ones(b)
            i = 0

            encoder_out = model.encoder(
                sample['net_input']['src_tokens'],
                src_lengths=sample['net_input']['src_lengths'],
                att_masks=sample['net_input']['att_masks'],
                patch_images=sample['net_input']['patch_images'],
                patch_masks=sample['net_input']['patch_masks'],
                token_embeddings=None,
                return_all_hiddens=False,
                sample_patch_num=None
            )

            while i < max_len and unfinish_flag.any():
                prev_output_tokens_11_tensor = torch.tensor(np.array(prev_output_token_11)).to(img.device).long()
                prev_output_tokens_12_tensor = torch.tensor(np.array(prev_output_token_12)).to(img.device).long()
                prev_output_tokens_21_tensor = torch.tensor(np.array(prev_output_token_21)).to(img.device).long()
                prev_output_tokens_22_tensor = torch.tensor(np.array(prev_output_token_22)).to(img.device).long()
                delta_x1_tensor = torch.tensor(np.array(delta_x1)).to(img.device)
                delta_x2_tensor = torch.tensor(np.array(delta_x2)).to(img.device)
                delta_y1_tensor = torch.tensor(np.array(delta_y1)).to(img.device)
                delta_y2_tensor = torch.tensor(np.array(delta_y2)).to(img.device)

                net_output = model.decoder(
                    prev_output_tokens_11_tensor,
                    prev_output_tokens_12_tensor,
                    prev_output_tokens_21_tensor,
                    prev_output_tokens_22_tensor,
                    delta_x1_tensor,
                    delta_y1_tensor,
                    delta_x2_tensor,
                    delta_y2_tensor,
                    code_masks=None,
                    encoder_out=encoder_out,
                    features_only=False,
                    alignment_layer=None,
                    alignment_heads=None,
                    src_lengths=sample['net_input']['src_lengths'],
                    return_all_hiddens=False
                )

                cls_output = net_output[0]
                cls_type = torch.argmax(cls_output, 2)
                reg_output = net_output[1]
                for j in range(b):
                    if unfinish_flag[j] == 1:  # prediction is not finished
                        cls_j = cls_type[j, i].item()
                        if cls_j == COO or (cls_j == EOS and i < min_len):
                            output_j_x, output_j_y = reg_output[j, i].cpu().numpy()
                            output_j_x = min(output_j_x, 1)
                            output_j_y = min(output_j_y, 1)

                            gen_out[j].extend([output_j_x, output_j_y])

                            output_j_x = output_j_x * (n_bins - 1)
                            output_j_y = output_j_y * (n_bins - 1)

                            output_j_x_floor = math.floor(output_j_x)
                            output_j_y_floor = math.floor(output_j_y)
                            output_j_x_ceil = math.ceil(output_j_x)
                            output_j_y_ceil = math.ceil(output_j_y)

                            # tokenization
                            prev_output_token_11[j].append(output_j_x_floor * n_bins + output_j_y_floor + 4)
                            prev_output_token_12[j].append(output_j_x_floor * n_bins + output_j_y_ceil + 4)
                            prev_output_token_21[j].append(output_j_x_ceil * n_bins + output_j_y_floor + 4)
                            prev_output_token_22[j].append(output_j_x_ceil * n_bins + output_j_y_ceil + 4)

                            delta_x = output_j_x - output_j_x_floor
                            delta_y = output_j_y - output_j_y_floor

                        elif cls_j == SEP:
                            gen_out[j].append(2)  # insert 2 indicating separator tokens
                            prev_output_token_11[j].append(sep_index)
                            prev_output_token_12[j].append(sep_index)
                            prev_output_token_21[j].append(sep_index)
                            prev_output_token_22[j].append(sep_index)
                            delta_x = 0
                            delta_y = 0
                        else:  # eos is predicted and i >= min_len
                            unfinish_flag[j] = 0
                            gen_out[j].append(-1)
                            prev_output_token_11[j].append(2)  # 2 is eos token
                            prev_output_token_12[j].append(2)  # 2 is eos token
                            prev_output_token_21[j].append(2)  # 2 is eos token
                            prev_output_token_22[j].append(2)  # 2 is eos token
                            delta_x = 0
                            delta_y = 0
                    else:  # prediction is finished
                        gen_out[j].append(-1)
                        prev_output_token_11[j].append(1)  # 1 is padding token
                        prev_output_token_12[j].append(1)
                        prev_output_token_21[j].append(1)
                        prev_output_token_22[j].append(1)
                        delta_x = 0
                        delta_y = 0
                    delta_x1[j].append(delta_x)
                    delta_y1[j].append(delta_y)
                    delta_x2[j].append(1 - delta_x)
                    delta_y2[j].append(1 - delta_y)
                i += 1
        print("inference step: ", i)
        return gen_out

