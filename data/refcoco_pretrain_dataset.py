# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import logging
import warnings

import numpy as np
import torch
import base64
import utils.transforms as T
import math
import os
from PIL import Image, ImageFile

from data import data_utils
from data.base_dataset import BaseDataset
from bert.tokenization_bert import BertTokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RefcocoPretrainDataset(BaseDataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=80,
        max_tgt_length=30,
        patch_image_size=512,
        imagenet_default_mean_and_std=False,
        num_bins=1000,
        max_image_size=512,
        image_path="../../datasets/images"
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins
        self.image_path = image_path

        if imagenet_default_mean_and_std:
            mean = IMAGENET_DEFAULT_MEAN
            std = IMAGENET_DEFAULT_STD
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]

        # for positioning
        self.positioning_transform = T.Compose([
            T.RandomResize([patch_image_size], max_size=patch_image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std, max_image_size=max_image_size)
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __getitem__(self, index):
        uniq_id, img_file, text, region_coord = self.dataset[index]

        img_path = os.path.join(self.image_path, img_file)
        image = Image.open(img_path).convert("RGB")

        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        x0, y0, x1, y1 = region_coord.strip().split(',')
        region = torch.tensor([float(x0), float(y0), float(x1), float(y1)])
        boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
        boxes_target["labels"] = np.array([0])
        boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])

        patch_image, patch_boxes = self.positioning_transform(image, boxes_target)
        resize_h, resize_w = patch_boxes["size"][0], patch_boxes["size"][1]
        patch_mask = torch.tensor([True])

        quant_box = [patch_boxes["boxes"][0][i] * (self.num_bins - 1) for i in range(4)]
        quant_box = np.array(quant_box).reshape(2, 2)

        quant_box11 = [[math.floor(p[0]), math.floor(p[1])] for p in quant_box]
        quant_box21 = [[math.ceil(p[0]), math.floor(p[1])] for p in quant_box]
        quant_box12 = [[math.floor(p[0]), math.ceil(p[1])] for p in quant_box]
        quant_box22 = [[math.ceil(p[0]), math.ceil(p[1])] for p in quant_box]


        # compute linear interpolation coefficient (0 for bos token)
        delta_x1 = torch.tensor([0] + [p[0] - math.floor(p[0]) for p in quant_box])
        delta_y1 = torch.tensor([0] + [p[1] - math.floor(p[1]) for p in quant_box])
        delta_x2 = 1 - delta_x1
        delta_y2 = 1 - delta_y1

        region_coord11 = " ".join([f"<bin_{int(p[0])}_{int(p[1])}>" for p in quant_box11])
        region_coord21 = " ".join([f"<bin_{int(p[0])}_{int(p[1])}>" for p in quant_box21])
        region_coord12 = " ".join([f"<bin_{int(p[0])}_{int(p[1])}>" for p in quant_box12])
        region_coord22 = " ".join([f"<bin_{int(p[0])}_{int(p[1])}>" for p in quant_box22])

        src_caption = self.pre_caption(text, self.max_src_length)

        prompt = ' which region does the text " {} " describe?'.format(src_caption)

        # tgt for input
        tgt_item11 = self.encode_text(region_coord11, use_bpe=False)
        tgt_item12 = self.encode_text(region_coord12, use_bpe=False)
        tgt_item21 = self.encode_text(region_coord21, use_bpe=False)
        tgt_item22 = self.encode_text(region_coord22, use_bpe=False)

        # tgt for output
        tgt_box = torch.reshape(patch_boxes["boxes"][0], (2, 2))
        target_item = torch.cat([tgt_box, torch.tensor([[1, 1]])], dim=0)  # [1, 1] is padding token for eos

        #target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item11 = torch.cat([self.bos_item, tgt_item11])
        prev_output_item12 = torch.cat([self.bos_item, tgt_item12])
        prev_output_item21 = torch.cat([self.bos_item, tgt_item21])
        prev_output_item22 = torch.cat([self.bos_item, tgt_item22])
        example = {
            "id": uniq_id,
            "source": prompt,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens_11": prev_output_item11,
            "prev_output_tokens_12": prev_output_item12,
            "prev_output_tokens_21": prev_output_item21,
            "prev_output_tokens_22": prev_output_item22,
            "delta_x1": delta_x1,
            "delta_y1": delta_y1,
            "delta_x2": delta_x2,
            "delta_y2": delta_y2,
            "w_resize_ratio": resize_w / w,
            "h_resize_ratio": resize_h / h,
            "region_coord": region,
            "token_type": torch.tensor([0, 0, 2])
        }
        return example

    def collate(self, samples, pad_idx, eos_idx):
        if len(samples) == 0:
            return {}

        def merge(key):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                pad_idx,
                eos_idx=eos_idx,
            )

        id = np.array([s["id"] for s in samples])
        captions = [s["source"] for s in samples]
        tokenized = self.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt")
        src_tokens = tokenized["input_ids"]
        att_masks = tokenized["attention_mask"]
        src_lengths = torch.LongTensor(att_masks.ne(0).long().sum())

        patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
        patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

        w_resize_ratios = torch.stack([s["w_resize_ratio"] for s in samples], dim=0)
        h_resize_ratios = torch.stack([s["h_resize_ratio"] for s in samples], dim=0)

        delta_x1 = torch.stack([s["delta_x1"] for s in samples], dim=0)
        delta_y1 = torch.stack([s["delta_y1"] for s in samples], dim=0)
        delta_x2 = torch.stack([s["delta_x2"] for s in samples], dim=0)
        delta_y2 = torch.stack([s["delta_y2"] for s in samples], dim=0)

        region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

        target = merge("target")
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        prev_output_tokens_11 = merge("prev_output_tokens_11")
        prev_output_tokens_12 = merge("prev_output_tokens_12")
        prev_output_tokens_21 = merge("prev_output_tokens_21")
        prev_output_tokens_22 = merge("prev_output_tokens_22")

        token_type = merge("token_type")

        batch = {
            "id": id,
            "nsentences": len(samples),
            "ntokens": ntokens,
            "net_input": {
                "src_tokens": src_tokens,
                "src_lengths": src_lengths,
                "att_masks": att_masks,
                "patch_images": patch_images,
                "patch_masks": patch_masks,
                "prev_output_tokens_11": prev_output_tokens_11,
                "prev_output_tokens_12": prev_output_tokens_12,
                "prev_output_tokens_21": prev_output_tokens_21,
                "prev_output_tokens_22": prev_output_tokens_22,
                "delta_x1": delta_x1,
                "delta_y1": delta_y1,
                "delta_x2": delta_x2,
                "delta_y2": delta_y2
            },
            "target": target,
            "token_type": token_type,
            "w_resize_ratios": w_resize_ratios,
            "h_resize_ratios": h_resize_ratios,
            "region_coords": region_coords
        }

        return batch

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.
        Args:
            samples (List[dict]): samples to collate
        Returns:
            dict: a mini-batch containing the data of the task
        """
        return self.collate(samples, pad_idx=self.pad, eos_idx=self.eos)