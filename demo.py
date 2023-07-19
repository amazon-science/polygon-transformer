import torch
import numpy as np
from fairseq import utils,tasks
from utils.checkpoint_utils import load_model_ensemble_and_task
from utils.eval_utils import eval_step
from tasks.refcoco import RefcocoTask
from models.polyformer import PolyFormerModel
from PIL import Image
import cv2
import math
from skimage import draw


tasks.register_task('refcoco', RefcocoTask)

# turn on cuda if GPU is available
use_cuda = torch.cuda.is_available()
# use fp16 only when GPU is available
use_fp16 = True

# Load pretrained ckpt & config
overrides={"bpe_dir":"utils/BPE"}
models, cfg, task = load_model_ensemble_and_task(
        utils.split_paths('weights/polyformer_l_refcocog.pt'),
        arg_overrides=overrides
    )

cfg.common.seed = 7
cfg.generation.beam = 5
cfg.generation.min_len = 12
cfg.generation.max_len_a = 0
cfg.generation.max_len_b = 420
cfg.generation.no_repeat_ngram_size = 3
cfg.task.patch_image_size = 512

from bert.tokenization_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Fix seed for stochastic decoding
if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)


# Move models to GPU
for model in models:
    model.eval()
    if use_fp16:
        model.half()
    if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
        model.cuda()
    model.prepare_for_inference_(cfg)

# Initialize generator
generator = task.build_generator(models, cfg.generation)


# Image transform
from torchvision import transforms
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])

# Text preprocess
bos_item = torch.LongTensor([task.src_dict.bos()])
eos_item = torch.LongTensor([task.src_dict.eos()])
pad_idx = task.src_dict.pad()


# Construct input for refcoco task
patch_image_size = cfg.task.patch_image_size
def construct_sample(image: Image, text: str):
    w, h = image.size
    w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
    h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    
    prompt = ' which region does the text " {} " describe?'.format(text)
    tokenized = tokenizer.batch_encode_plus([prompt], padding="longest", return_tensors="pt")
    src_tokens = tokenized["input_ids"]
    att_masks = tokenized["attention_mask"]
    src_lengths = torch.LongTensor(att_masks.ne(0).long().sum())
    
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "att_masks": att_masks,
            "patch_images": patch_image,
            "patch_masks": patch_mask,
        },
        "w_resize_ratios": w_resize_ratio,
        "h_resize_ratios": h_resize_ratio,
        "region_coords": torch.randn(1, 4),
        "label": np.zeros((512,512)),
        "poly": 'None',
        "text": text
    }
    return sample

# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


from io import BytesIO
import base64
import re

def pre_caption(caption):
    caption = caption.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ').replace('<person>', 'person')

    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n')
    caption = caption.strip(' ')
    return caption


def convert_pts(coeffs):
    pts = []
    for i in range(len(coeffs) // 2):
        pts.append([coeffs[2 * i + 1], coeffs[2 * i]])  # y, x
    return np.array(pts, np.int32)

def get_mask_from_codes(codes, img_size):
    masks = [np.zeros(img_size)]
    for code in codes:
        mask = draw.polygon2mask(img_size, convert_pts(code))
        mask = np.array(mask, np.uint8)
        masks.append(mask)
    mask = sum(masks)
    mask = mask > 0
    return mask.astype(np.uint8)


def overlay_predictions(img, mask=None, polygons=None, bbox=None, color_box=(0, 255, 0), color_mask=[255, 102, 102], color_poly=[255, 0, 0], thickness=3, radius=6):
    overlayed = img.copy()
    if bbox is not None:
        overlayed = draw_bbox(overlayed, bbox, color=color_box, thickness=thickness)
    if mask is not None:
        overlayed = overlay_davis(overlayed, mask, colors=[[0, 0, 0], color_mask])
    if polygons is not None:
        overlayed = plot_polygons(overlayed, polygons, color=color_poly, radius=radius)
    return overlayed


def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 102, 102]], cscale=1, alpha=0.4):  # [255, 178, 102] orange [102, 178, 255] red
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    h_i, w_i = image.shape[0:2]
    h_m, w_m = mask.shape[0:2]
    if h_i != h_m:
        mask = cv2.resize(mask, [h_i, w_i], interpolation=cv2.INTER_NEAREST)
    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

    return im_overlay.astype(image.dtype)


def draw_bbox(img, box, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = box
    return cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)

def plot_polygons(img, polygons, color=(255, 0, 0), radius=7):
    for polygon in polygons:
        if len(polygon) > 0:
            polygon = np.reshape(polygon[:len(polygon)-len(polygon)%2], (len(polygon)//2, 2)).astype(np.int16)
            for i, point in enumerate(polygon):
                img = cv2.circle(img, point, radius, color, thickness=-1)
            img = cv2.circle(img, polygon[0], radius, color, thickness=-1)
    return img

def plot_arrow(img, polygons, color=(128, 128, 128), thickness=3, tip_length=0.3):
    for polygon in polygons:
        if len(polygon) > 0:
            polygon = np.reshape(polygon[:len(polygon)-len(polygon)%2], (len(polygon)//2, 2)).astype(np.int16)
            for i, point in enumerate(polygon):
                if i > 0: 
                    img = cv2.arrowedLine(img, polygon[i-1], point, color, thickness=thickness, tipLength=tip_length)  
    return img

def downsample_polygon(polygon, ds_rate=25):
    points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    points = points[::ds_rate]
    return list(points.flatten())


def downsample_polygons(polygons, ds_rate=25):
    polygons_ds = []
    for polygon in polygons:
        polygons_ds.append(downsample_polygon(polygon, ds_rate))
    return polygons_ds



def visual_grounding(image, text):
    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image, text.lower())
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

    with torch.no_grad():
        if isinstance(models, list):
            model = models[0]
        min_len = 6
        max_len = 210
        model.eval()
        img = sample["net_input"]["patch_images"]
        b = img.shape[0]
        prev_output_token_11 = [[0] for _ in range(b)]
        prev_output_token_12 = [[0] for _ in range(b)]
        prev_output_token_21 = [[0] for _ in range(b)]
        prev_output_token_22 = [[0] for _ in range(b)]
        delta_x1 = [[0] for _ in range(b)]
        delta_y1 = [[0] for _ in range(b)]
        delta_x2 = [[1] for _ in range(b)]
        delta_y2 = [[1] for _ in range(b)]

        gen_out = [[] for _ in range(b)]

        n_bins = 64

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
        attn_masks = []
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
            reg_output = net_output[1].squeeze(-1)
            attn = net_output[2]['attn']
            attn_arrays = [att.detach().cpu().numpy() for att in attn]
            attn_arrays = np.concatenate(attn_arrays, 0)
            attn_arrays = np.mean(attn_arrays, 0)
            attn_arrays = attn_arrays[i, :256].reshape(16, 16)
            h, w = image.size
            attn_mask = cv2.resize(attn_arrays.astype(np.float32), (h, w))
            attn_masks.append(attn_mask)
            
            for j in range(b):
                if unfinish_flag[j] == 1:  # prediction is not finished
                    cls_j = cls_type[j, i].item()
                    if cls_j == 0 or (cls_j == 2 and i < min_len):  # 0 for coordinate tokens; 2 for eos
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

                        # convert to token
                        prev_output_token_11[j].append(output_j_x_floor * n_bins + output_j_y_floor + 4)
                        prev_output_token_12[j].append(output_j_x_floor * n_bins + output_j_y_ceil + 4)
                        prev_output_token_21[j].append(output_j_x_ceil * n_bins + output_j_y_floor + 4)
                        prev_output_token_22[j].append(output_j_x_ceil * n_bins + output_j_y_ceil + 4)

                        delta_x = output_j_x - output_j_x_floor
                        delta_y = output_j_y - output_j_y_floor
                    elif cls_j == 1:  # 1 for separator tokens
                        gen_out[j].append(2)  # insert 2 indicating separator tokens
                        prev_output_token_11[j].append(3)
                        prev_output_token_12[j].append(3)
                        prev_output_token_21[j].append(3)
                        prev_output_token_22[j].append(3)
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

    hyps = []
    hyps_det = []
    n_poly_pred = []
    b = len(gen_out)
    for i in range(b):
        gen_out_i = np.array(gen_out[i])
        gen_out_i = gen_out_i[gen_out_i != -1]  # excluding eos and padding indices


        gen_out_i_det = gen_out_i[:4]
        w, h = image.size
        gen_out_i_det[::2] *= w
        gen_out_i_det[1::2] *= h

        polygons_pred = gen_out_i[4:]
        polygons_pred = np.append(polygons_pred, [2])
        size = len(polygons_pred)
        idx_list = [idx for idx, val in
                    enumerate(polygons_pred) if val == 2]   # 2 indicates separator token

        polygons_pred[::2] *= w
        polygons_pred[1::2] *= h
        if len(idx_list) > 0:   # multiple polygons
            polygons = []
            pred_idx = 0
            for idx in idx_list:
                cur_idx = idx
                if pred_idx == cur_idx or pred_idx == size:
                    pass
                else:
                    polygons.append(polygons_pred[pred_idx: cur_idx])
                pred_idx = cur_idx + 1
        else:
            polygons = [polygons_pred]

        n_poly_pred.append(len(polygons))
        hyps.append(polygons)
        hyps_det.append(gen_out_i_det)
        

    pred_mask = get_mask_from_codes(hyps[0], (h, w))
    pred_overlayed = overlay_predictions(np.asarray(image), pred_mask, hyps[0], hyps_det[0])

    return pred_overlayed, np.array(pred_mask*255, dtype=np.uint8)


