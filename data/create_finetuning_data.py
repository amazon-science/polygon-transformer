from refer.refer import REFER
import numpy as np
from PIL import Image
import random
import os
from tqdm import tqdm

import pickle
from poly_utils import is_clockwise, revert_direction, check_length, reorder_points, \
    approximate_polygons, interpolate_polygons, image_to_base64, polygons_to_string


max_length = 400

data_root = './refer/data'
datasets = ['refcoco', 'refcoco+', 'refcocog']

image_dir = './datasets/images/mscoco/train2014'
val_test_files = pickle.load(open("data/val_test_files.p", "rb"))

combined_train_data = []

for dataset in datasets:
    if dataset == 'refcoco':
        splits = ['train', 'val', 'testA', 'testB']
        splitBy = 'unc'
    elif dataset == 'refcoco+':
        splits = ['train', 'val', 'testA', 'testB']
        splitBy = 'unc'
    elif dataset == 'refcocog':
        splits = ['train', 'val']
        splitBy = 'umd'

    save_dir = f'datasets/finetune/{dataset}'
    os.makedirs(save_dir, exist_ok=True)
    for split in splits:
        num_pts = []
        max_num_pts = 0
        file_name = os.path.join(save_dir, f"{dataset}_{split}.tsv")
        print("creating ", file_name)

        uniq_ids = []
        image_ids = []
        sents = []
        coeffs_strings = []
        img_strings = []

        writer = open(file_name, 'w')
        refer = REFER(data_root, dataset, splitBy)

        ref_ids = refer.getRefIds(split=split)

        for this_ref_id in tqdm(ref_ids):
            this_img_id = refer.getImgIds(this_ref_id)
            this_img = refer.Imgs[this_img_id[0]]
            fn = this_img['file_name']
            img_id = fn.split(".")[0].split("_")[-1]

            # load image
            img = Image.open(os.path.join(image_dir, this_img['file_name'])).convert("RGB")

            # convert image to string
            img_base64 = image_to_base64(img, format='jpeg')

            # load mask
            ref = refer.loadRefs(this_ref_id)
            ref_mask = np.array(refer.getMask(ref[0])['mask'])
            annot = np.zeros(ref_mask.shape)
            annot[ref_mask == 1] = 1  # 255
            annot_img = Image.fromarray(annot.astype(np.uint8), mode="P")
            annot_base64 = image_to_base64(annot_img, format='png')

            polygons = refer.getPolygon(ref[0])['polygon']

            polygons_processed = []
            for polygon in polygons:
                # make the polygon clockwise
                if not is_clockwise(polygon):
                    polygon = revert_direction(polygon)

                # reorder the polygon so that the first vertex is the one closest to image origin
                polygon = reorder_points(polygon)
                polygons_processed.append(polygon)

            polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
            polygons_interpolated = interpolate_polygons(polygons)

            polygons = approximate_polygons(polygons, 5, max_length)

            pts_string = polygons_to_string(polygons)
            pts_string_interpolated = polygons_to_string(polygons_interpolated)

            # load box
            box = refer.getRefBox(this_ref_id)  # x,y,w,h
            x, y, w, h = box
            box_string = f'{x},{y},{x + w},{y + h}'

            max_num_pts = max(max_num_pts, check_length(polygons))

            num_pts.append(check_length(polygons))
            # load text
            ref_sent = refer.Refs[this_ref_id]
            for i, (sent, sent_id) in enumerate(zip(ref_sent['sentences'], ref_sent['sent_ids'])):
                uniq_id = f"{this_ref_id}_{i}"
                instance = '\t'.join(
                    [uniq_id, str(this_img_id[0]), sent['sent'], box_string, pts_string, img_base64, annot_base64,
                     pts_string_interpolated]) + '\n'
                writer.write(instance)

                if img_id not in val_test_files and split == 'train':  # filtered out val/test files
                    combined_train_data.append(instance)
        writer.close()

random.shuffle(combined_train_data)
file_name = os.path.join("datasets/finetune/refcoco+g_train_shuffled.tsv")
print("creating ", file_name)
writer = open(file_name, 'w')
writer.writelines(combined_train_data)
writer.close()




