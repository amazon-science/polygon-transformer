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
from PIL import Image, ImageFile

from data import data_utils
from data.base_dataset import BaseDataset
from bert.tokenization_bert import BertTokenizer
from data.poly_utils import string_to_polygons, downsample_polygons, polygons_to_string, points_to_token_string
import cv2

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class RefcocoDataset(BaseDataset):
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
        max_image_size=512
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.patch_image_size = patch_image_size
        self.num_bins = num_bins

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
        data = self.dataset[index]
        if len(data) == 7:
            uniq_id, base64_str, seg64_str, text, poly_original, region_coord, poly_interpolated = data
            train = True
        else:
            uniq_id, base64_str, seg64_str, text, poly, region_coord = data
            train = False

        # load image and segmentation labels
        image = Image.open(BytesIO(base64.urlsafe_b64decode(base64_str))).convert("RGB")
        label = Image.open(BytesIO(base64.urlsafe_b64decode(seg64_str)))
        label = np.asarray(label)
        label = cv2.resize(label, [self.patch_image_size, self.patch_image_size], interpolation=cv2.INTER_NEAREST)

        w, h = image.size
        patch_image = self.positioning_transform(image, target=None)
        resize_h = self.patch_image_size
        resize_w = self.patch_image_size
        patch_mask = torch.tensor([True])

        if train:
            prob = np.random.uniform()
            if prob < 0.5:
                polygons_interpolated = string_to_polygons(poly_interpolated)
                ds_rate = np.random.randint(25, 41)
                polygons_augmented = downsample_polygons(polygons_interpolated, ds_rate)
                poly = polygons_to_string(polygons_augmented)
            else:
                poly = poly_original

        polygons = string_to_polygons(poly)
        polygons_scaled = []
        for polygon in polygons:
            n_point = len(polygon) // 2
            scale = np.concatenate([np.array([w, h]) for _ in range(n_point)], 0)
            polygon = polygon / scale
            polygon = polygon.reshape(n_point, 2)
            polygons_scaled.append(polygon)

        x0, y0, x1, y1 = region_coord.strip().split(',')
        region_points = [float(x0), float(y0), float(x1), float(y1)]
        region = np.array(region_points)

        region_points = region_points / np.array([w, h, w, h])  # scaled to [0,1]
        region_points = torch.tensor(region_points.reshape(2, 2))

        quant_box = region_points * (self.num_bins - 1)
        quant_box11 = [[math.floor(p[0]), math.floor(p[1])] for p in quant_box]
        quant_box21 = [[math.ceil(p[0]), math.floor(p[1])] for p in quant_box]
        quant_box12 = [[math.floor(p[0]), math.ceil(p[1])] for p in quant_box]
        quant_box22 = [[math.ceil(p[0]), math.ceil(p[1])] for p in quant_box]

        quant_poly = [poly * (self.num_bins - 1) for poly in polygons_scaled]
        quant_poly11 = [[[math.floor(p[0]), math.floor(p[1])] for p in poly] for poly in quant_poly]
        quant_poly21 = [[[math.ceil(p[0]), math.floor(p[1])] for p in poly] for poly in quant_poly]
        quant_poly12 = [[[math.floor(p[0]), math.ceil(p[1])] for p in poly] for poly in quant_poly]
        quant_poly22 = [[[math.ceil(p[0]), math.ceil(p[1])] for p in poly] for poly in quant_poly]

        region_coord11, _ = points_to_token_string(quant_box11, quant_poly11)
        region_coord21, _ = points_to_token_string(quant_box21, quant_poly21)
        region_coord12, _ = points_to_token_string(quant_box12, quant_poly12)
        region_coord22, token_type = points_to_token_string(quant_box22, quant_poly22)

        # compute bilinear interpolation coefficient
        delta_x1 = [0] + [p[0] - math.floor(p[0]) for p in quant_box]  # [0] for bos token
        for polygon in quant_poly:
            delta = [poly_point[0] - math.floor(poly_point[0]) for poly_point in polygon]
            delta_x1.extend(delta)
            delta_x1.extend([0])  # for separator token
        delta_x1 = delta_x1[:-1]  # there is no separator token in the end
        delta_x1 = torch.tensor(delta_x1)
        delta_x2 = 1 - delta_x1

        delta_y1 = [0] + [p[1] - math.floor(p[1]) for p in quant_box]  # [0] for bos token
        for polygon in quant_poly:
            delta = [poly_point[1] - math.floor(poly_point[1]) for poly_point in polygon]
            delta_y1.extend(delta)
            delta_y1.extend([0])  # for separator token
        delta_y1 = delta_y1[:-1]  # there is no separator token in the end
        delta_y1 = torch.tensor(delta_y1)
        delta_y2 = 1 - delta_y1

        token_type.append(2)  # 2 for eos token

        src_caption = self.pre_caption(text, self.max_src_length)

        prompt = ' which region does the text " {} " describe?'.format(src_caption)

        # tgt for input
        tgt_item11 = self.encode_text(region_coord11, use_bpe=False)
        tgt_item12 = self.encode_text(region_coord12, use_bpe=False)
        tgt_item21 = self.encode_text(region_coord21, use_bpe=False)
        tgt_item22 = self.encode_text(region_coord22, use_bpe=False)

        # tgt for output
        target_item = region_points
        for poly in polygons_scaled:
            target_item = torch.cat([target_item, torch.tensor(poly), torch.tensor([[0, 0]])], dim=0)  # [0, 0] is padding token for separator and eos

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
            "w_resize_ratio": torch.tensor(resize_w / w),
            "h_resize_ratio": torch.tensor(resize_h / h),
            "region_coord": torch.tensor(region),
            "token_type": torch.tensor(token_type),
            "w": torch.tensor(w),
            "h": torch.tensor(h),
            "label": label,
            "n_poly": len(polygons),
            "text": src_caption
        }
        return example

    def collate(self, samples, pad_idx, eos_idx):
        if len(samples) == 0:
            return {}

        def merge(key, padding_item):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                padding_item,
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

        delta_x1 = merge("delta_x1", 0)
        delta_y1 = merge("delta_y1", 0)
        delta_x2 = merge("delta_x2", 1)
        delta_y2 = merge("delta_y2", 1)

        region_coords = torch.stack([s['region_coord'] for s in samples], dim=0)

        target = merge("target", pad_idx)
        tgt_lengths = torch.LongTensor([s["target"].shape[0] for s in samples])
        ntokens = tgt_lengths.sum().item()

        prev_output_tokens_11 = merge("prev_output_tokens_11", pad_idx)
        prev_output_tokens_12 = merge("prev_output_tokens_12", pad_idx)
        prev_output_tokens_21 = merge("prev_output_tokens_21", pad_idx)
        prev_output_tokens_22 = merge("prev_output_tokens_22", pad_idx)

        token_type = merge("token_type", -1)
        w = torch.stack([s["w"] for s in samples], dim=0)
        h = torch.stack([s["h"] for s in samples], dim=0)
        n_poly = [s['n_poly'] for s in samples]

        labels = np.stack([sample['label'] for sample in samples], 0)
        text = [s["text"] for s in samples]
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
            "w_resize_ratios": w_resize_ratios,
            "h_resize_ratios": h_resize_ratios,
            "region_coords": region_coords,
            "label": labels,
            "token_type": token_type,
            "w": w,
            "h": h,
            "n_poly": n_poly,
            "text": text
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