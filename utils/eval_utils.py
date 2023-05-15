# ------------------------------------------------------------------------
# Modified from OFA (https://github.com/OFA-Sys/OFA)
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.
# ------------------------------------------------------------------------
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from itertools import chain
import os
import torch
import torch.distributed as dist
import numpy as np
from skimage import draw
from PIL import Image
from utils.vis_utils import overlay_predictions
from torchvision.utils import save_image

SMOOTH = 1e-6


def check_length(polygons):
    length = 0
    for polygon in polygons:
        length += len(polygon)
    return length


def eval_refcoco(task, generator, models, sample, **kwargs):
    def _computeIoU(pred_seg, gd_seg):
        I = np.sum(np.logical_and(pred_seg, gd_seg))
        U = np.sum(np.logical_or(pred_seg, gd_seg))
        return I, U

    def _calculate_ap_score(hyps, refs, thresh=0.5):
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

    def convert_pts(coeffs):
        pts = []
        for i in range(len(coeffs) // 2):
            pts.append([coeffs[2 * i + 1], coeffs[2 * i]])  # y, x
        return np.array(pts, np.int32)

    def get_mask_from_codes(codes, img_size):
        masks = [np.zeros(img_size)]
        for code in codes:
            if len(code) > 0:
                try:
                    mask = draw.polygon2mask(img_size, convert_pts(code))
                    mask = np.array(mask, np.uint8)
                except:
                    mask = np.zeros(img_size)
                masks.append(mask)
        mask = sum(masks)
        mask = mask > 0
        return mask.astype(np.uint8)

    def _calculate_score(hyps, hyps_det, refs, sample, n_poly_pred, n_poly_gt, vis=True, vis_dir=None):
        if vis:
            os.makedirs(vis_dir, exist_ok=True)

        def compute_jf(pred_mask, gt_mask):
            I, U = _computeIoU(pred_mask, gt_mask)
            if U == 0:
                this_iou = 0.0
            else:
                this_iou = I * 1.0 / U

            prec = (I + SMOOTH) / (pred_mask.sum() + SMOOTH)
            rec = (I + SMOOTH) / (gt_mask.sum() + SMOOTH)
            this_f = 2 * prec * rec / (prec + rec)
            return this_iou, this_f, I, U

        IoU = []
        F_score = []
        cum_I = []
        cum_U = []
        bboxes = hyps_det
        b = len(hyps)
        bboxes = torch.tensor(np.stack(bboxes, 0))
        bboxes = bboxes.to(sample['w_resize_ratios'].device)
        ap_scores = _calculate_ap_score(bboxes.float(), sample['region_coords'].float())
        for i in range(b):
            hyps_i = hyps[i]
            gt_mask = refs[i]
            pred_mask = get_mask_from_codes(hyps_i, gt_mask.shape[0:2])
            this_iou, this_f, this_I, this_U = compute_jf(pred_mask, gt_mask)
            IoU.append(this_iou)
            F_score.append(this_f)
            cum_I.append(this_I)
            cum_U.append(this_U)

            if vis:
                def pre_caption(caption):
                    import re
                    caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace(
                        '<person>',
                        'person')
                    caption = re.sub(
                        r"\s{2,}",
                        ' ',
                        caption,
                    )
                    caption = caption.rstrip('\n')
                    return caption

                gt_box = sample['region_coords'][i].cpu().numpy()
                pred_box = bboxes[i].cpu().numpy()
                pred_box[::2] *= sample['w_resize_ratios'][i].cpu().numpy()
                pred_box[1::2] *= sample['h_resize_ratios'][i].cpu().numpy()
                gt_box[::2] *= sample['w_resize_ratios'][i].cpu().numpy()
                gt_box[1::2] *= sample['h_resize_ratios'][i].cpu().numpy()
                uniq_id = sample["id"][i]
                text = sample["text"][i]
                text = pre_caption(text)
                img = sample["net_input"]['patch_images'][i]
                img = (img + 1) / 2
                img_ndarray = img.permute(1, 2, 0).cpu().numpy() * 255
                img_ndarray = img_ndarray.astype(np.uint8)

                gt_overlayed_fn = f"{uniq_id}_{text}_gt_overlayed.png"
                pred_overlayed_fn = f"{uniq_id}_{text}_pred_overlayed.png"

                pred_overlayed = overlay_predictions(img_ndarray, pred_mask, hyps_i, pred_box)
                gt_overlayed = overlay_predictions(img_ndarray, gt_mask, None, gt_box)

                pred_overlayed = Image.fromarray(pred_overlayed.astype(np.uint8))
                pred_overlayed.save(os.path.join(vis_dir, pred_overlayed_fn))
                gt_overlayed = Image.fromarray(gt_overlayed.astype(np.uint8))
                gt_overlayed.save(os.path.join(vis_dir, gt_overlayed_fn))

                img_fn = f"{uniq_id}_{text}.png"
                save_image(img, os.path.join(vis_dir, img_fn))

        return torch.tensor(IoU), torch.tensor(F_score), ap_scores, torch.tensor(cum_I), torch.tensor(cum_U)

    gen_out = task.inference_step(models, sample)
    hyps = []
    hyps_det = []
    n_poly_pred = []
    b = len(gen_out)
    poly_len = []
    for i in range(b):
        gen_out_i = np.array(gen_out[i])
        gen_out_i = gen_out_i[gen_out_i != -1]  # excluding eos and padding indices

        gen_out_i_det = gen_out_i[:4]
        gen_out_i_det[::2] *= sample['w'][i].cpu().numpy()
        gen_out_i_det[1::2] *= sample['h'][i].cpu().numpy()

        polygons_pred = gen_out_i[4:]
        polygons_pred = np.append(polygons_pred, [2])
        size = len(polygons_pred)
        idx_list = [idx for idx, val in
                    enumerate(polygons_pred) if val == 2]  # 2 indicates separator token

        polygons_pred *= task.cfg.patch_image_size
        # extract the sequence for each polygon
        polygons = []
        prev_idx = 0
        for idx in idx_list:
            cur_idx = idx
            if prev_idx == cur_idx or prev_idx == size:
                pass
            else:
                polygons.append(polygons_pred[prev_idx: cur_idx])
            prev_idx = cur_idx + 1

        poly_len.append(check_length(polygons))
        n_poly_pred.append(len(polygons))
        hyps.append(polygons)
        hyps_det.append(gen_out_i_det)
    gt = sample['label']
    results = [
        {"uniq_id": sample_id}
        for i, sample_id in enumerate(sample["id"].tolist())
    ]

    iou_scores, f_scores, ap_scores, cum_I, cum_U = _calculate_score(hyps, hyps_det, gt, sample, n_poly_pred,
                                                                     sample['n_poly'],
                                                                     vis=kwargs['vis'], vis_dir=kwargs['vis_dir'])
    result_dir = kwargs['result_dir']
    os.makedirs(result_dir, exist_ok=True)
    torch.save({"iou_scores": iou_scores, "ap_scores": ap_scores, "n_poly_pred": n_poly_pred,
                "n_poly_gt": sample['n_poly'], "poly_len": poly_len, "uniq_id": sample["id"]},
               os.path.join(result_dir, f'{sample["id"][0]}.pt'))

    return results, iou_scores, f_scores, ap_scores, cum_I, cum_U


def eval_step(task, generator, models, sample, **kwargs):
    if task.cfg._name == 'refcoco':
        return eval_refcoco(task, generator, models, sample, **kwargs)
    else:
        raise NotImplementedError


def merge_results(task, cfg, logger, score_cnt, score_sum, f_score_sum=None, ap_det_score_sum=None, prec_score_sum=None,
                  cum_I_sum=None, cum_U_sum=None, results=None):
    if task.cfg._name == 'image_gen':
        if cfg.distributed_training.distributed_world_size > 1:
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))
    else:
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(f_score_sum.data)
            dist.all_reduce(cum_I_sum.data)
            dist.all_reduce(cum_U_sum.data)
            for prec_score in prec_score_sum:
                dist.all_reduce(prec_score.data)
            dist.all_reduce(ap_det_score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            prec_list = [.5, .6, .7, .8, .9]
            txt = "sample_cnt: {}, mIoU score: {}, oIoU score: {}, ap det score: {}, f score: {}, J&F: {}\n".format(
                score_cnt, round(score_sum.item() / score_cnt.item(), 4),
                round(cum_I_sum.item() / cum_U_sum.item(), 4),
                round(ap_det_score_sum.item() / score_cnt.item(), 4),
                round(f_score_sum.item() / score_cnt.item(), 4),
                round((f_score_sum.item() + score_sum.item()) / (2 * score_cnt.item()), 4)
            )

            prec_txt = " ".join(
                [f"prec@{prec}: {round(prec_score.item() / score_cnt.item(), 4)}\n" for prec, prec_score in
                 zip(prec_list, prec_score_sum)])
            txt += prec_txt

            logger.info(txt)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_result.txt".format(cfg.dataset.gen_subset))
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(txt)

        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
            gather_results = list(chain(*gather_results)) if gather_results is not None else results
            with open(output_path, 'w') as fw:
                json.dump(gather_results, fw)
