import numpy as np
import cv2

def overlay_predictions(img, mask=None, polygons=None, bbox=None):
    overlayed = img.copy()
    if bbox is not None:
        overlayed = draw_bbox(overlayed, bbox)
    if mask is not None:
        overlayed = overlay_davis(overlayed, mask)
    if polygons is not None:
        overlayed = plot_polygons(overlayed, polygons)
    return overlayed


def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 102, 102]], cscale=1, alpha=0.4):
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

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)


def draw_bbox(img, box, color=(0, 255, 0), thickness=3):
    x1, y1, x2, y2 = box
    return cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness=thickness)


def plot_polygons(img, polygons, radius=3):
    for polygon in polygons:
        if len(polygon) > 0:
            polygon = np.reshape(polygon[:len(polygon)-len(polygon)%2], (len(polygon)//2, 2)).astype(np.int16)
            for i, point in enumerate(polygon):
                color = (255, 0, 0)
                #if i == 0:
                #    color = (0, 0, 255)
                img = cv2.circle(img, point, radius, color, thickness=-1)
    return img