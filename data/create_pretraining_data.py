import json
import os
from tqdm import tqdm
import random
import pickle

# set up image paths
imgsfile = dict(
    coco='mscoco/train2014',
    vg='visual-genome',
    saiaprtc12='saiaprtc12',
    flickr='flickr30k'
)

# load annotation files
f = open("datasets/annotations/instances.json")
print("Loading annotation file")
data = json.load(f)
f.close()

# load the validation and test image list of refcoco, refcoco+, and refcocog
val_test_files = pickle.load(open("data/val_test_files.p", "rb"))

# create result folder
os.makedirs("datasets/pretrain", exist_ok=True)

# generate training tsv file
train_instances = data['train']
tsv_filename = "datasets/pretrain/train_shuffled.tsv"
writer = open(tsv_filename, 'w')
print("generating ", tsv_filename)

lines = []
for i, data_i in enumerate(tqdm(train_instances)):
    data_source = data_i['data_source']
    image_id = data_i['image_id']
    bbox = data_i['bbox']
    expressions = data_i['expressions']
    height, width = data_i['height'], data_i['width']
    x, y, w, h = bbox
    box_string = f'{x},{y},{x + w},{y + h}'
    img_name = "COCO_train2014_%012d.jpg" if "coco" in data_source else "%d.jpg"
    img_name = img_name % image_id
    filepath = os.path.join(imgsfile[data_source], img_name)
    line = '\t'.join([str(i), expressions[0].replace('\n', ''), box_string, filepath]) + '\n'
    lines.append(line)

# shuffle the training set
random.shuffle(lines)

# write training tsv file
writer.writelines(lines)
writer.close()

# generate validation tsv files
val_sets = ['val_refcoco_unc', 'val_refcocoplus_unc', 'val_refcocog_umd', 'val_flickr30k', 'val_referitgame_berkeley']
for val_set in val_sets:
    val_instances = data[val_set]
    tsv_filename = f"datasets/pretrain/{val_set}.tsv"
    writer = open(tsv_filename, 'w')
    print("generating ", tsv_filename)

    lines = []
    for i, data_i in enumerate(tqdm(val_instances)):
        data_source = data_i['data_source']
        image_id = data_i['image_id']
        bbox = data_i['bbox']
        expressions = data_i['expressions']
        height, width = data_i['height'], data_i['width']
        x, y, w, h = bbox
        box_string = f'{x},{y},{x + w},{y + h}'
        img_name = "COCO_train2014_%012d.jpg" if "coco" in data_source else "%d.jpg"
        img_name = img_name % image_id
        filepath = os.path.join(imgsfile[data_source], img_name)
        line = '\t'.join([str(i), expressions[0].replace('\n', ''), box_string, filepath]) + '\n'
        lines.append(line)

    # write tsv file
    writer.writelines(lines)
    writer.close()
