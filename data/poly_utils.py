import re
import numpy as np
from itertools import groupby
from PIL import Image
import math
from math import ceil, floor
from skimage import draw
from random import sample
import base64
from io import BytesIO

convert = lambda text: int(text) if text.isdigit() else text.lower()
natrual_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]


def points_to_token_string(box, polygons):
    polygon_strings = []
    for polygon in polygons:
        polygon_string = " ".join([f"<bin_{int(p[0])}_{int(p[1])}>" for p in polygon])
        polygon_strings.append(polygon_string)
    polygon_string = " <separator> ".join(polygon_strings)
    box_string = " ".join([f"<bin_{int(p[0])}_{int(p[1])}>" for p in box])
    token_string = " ".join([box_string, polygon_string])

    token_type = []
    for token in token_string.split(" "):
        if "bin" in token:
            token_type.append(0)  # 0 for coordinate tokens
        else:
            token_type.append(1)  # 1 for separator tokens
    return token_string, token_type


def resize_binary_mask(array, new_size):
    image = Image.fromarray(array.astype(np.uint8) * 255)
    image = image.resize(new_size)
    return np.asarray(image).astype(np.bool_)


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))

    return rle


def revert_direction(poly):
    poly = np.array(poly).reshape(int(len(poly) / 2), 2)
    poly = poly[::-1, :]
    return list(poly.flatten())


def reorder_points(poly):
    poly = np.array(poly)
    xs = poly[::2]
    ys = poly[1::2]
    points = np.array(poly).reshape(int(len(poly) / 2), 2)
    start = np.argmin(xs ** 2 + ys ** 2)  # smallest distance to the origin
    poly_reordered = np.concatenate([points[start:], points[:start]], 0)
    return list(poly_reordered.flatten())


def convert_pts(coeffs):
    pts = []
    for i in range(len(coeffs) // 2):
        pts.append([coeffs[2 * i + 1], coeffs[2 * i]])  # y, x
    return np.array(pts, np.int32)


def get_mask_from_codes(codes, img_size):
    masks = [np.zeros(img_size)]
    for code in codes:
        if len(code) > 0:
            mask = draw.polygon2mask(img_size, convert_pts(code))
            mask = np.array(mask, np.uint8)
            masks.append(mask)
    mask = sum(masks)
    mask = mask > 0
    return mask.astype(np.uint8)


def is_clockwise(poly):
    n = len(poly) // 2
    xs = poly[::2]
    xs.append(xs[0])
    ys = poly[1::2]
    ys.append(ys[0])
    area = 0
    for i in range(n):
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[i + 1], ys[i + 1]
        area += (x2 - x1) * (y2 + y1)
    return area < 0


def close_polygon_contour(poly):
    poly = np.array(poly).reshape(int(len(poly) / 2), 2)
    x1, y1 = poly[0]
    x2, y2 = poly[-1]
    if x1 != x2:
        poly = np.concatenate([poly, [poly[0]]], 0)
    return list(poly.flatten())


def close_polygons_contour(polygons):
    polygons_closed = []
    for polygon in polygons:
        polygon_closed = close_polygon_contour(polygon)
        polygons_closed.append(polygon_closed)
    return polygons_closed


def image_to_base64(img, format):
    output_buffer = BytesIO()
    img.save(output_buffer, format=format)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    base64_str = str(base64_str, encoding='utf-8')
    return base64_str


def process_polygons(polygons, redirection=True, reorder=True, close=False):
    polygons_processed = []
    for polygon in polygons:
        if redirection and not is_clockwise(polygon):
            polygon = revert_direction(polygon)
        if reorder:
            polygon = reorder_points(polygon)
        if close:
            polygon = close_polygon_contour(polygon)
        polygons_processed.append(polygon)
    polygons = sorted(polygons_processed, key=lambda x: (x[0] ** 2 + x[1] ** 2, x[0], x[1]))
    return polygons


def string_to_polygons(pts_strings):
    pts_strings = pts_strings.split(" ")[:-1]
    polygons = []
    for pts_string in pts_strings:
        polygon = pts_string.split(",")
        polygon = [float(p) for p in polygon]
        polygons.append(polygon)
    return polygons


def downsample_polygon(polygon, ds_rate=25):
    points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    points = points[::ds_rate]
    return list(points.flatten())


def downsample_polygons(polygons, ds_rate=25):
    polygons_ds = []
    for polygon in polygons:
        polygons_ds.append(downsample_polygon(polygon, ds_rate))
    return polygons_ds


def check_length(polygons):
    length = 0
    for polygon in polygons:
        length += len(polygon)
    return length


def approximate_polygon(poly, tolerance=2):
    poly = np.array(poly).reshape(int(len(poly) / 2), 2)
    new_poly = [poly[0]]
    for i in range(1, len(poly)):
        x1, y1 = new_poly[-1]
        x2, y2 = poly[i]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist > tolerance:
            new_poly.append(poly[i])
    new_poly = np.array(new_poly)
    return list(new_poly.flatten())


def approximate_polygons(polys, tolerance=1.0, max_length=400):
    tol = tolerance
    while check_length(polys) > max_length:
        polys_new = []
        for poly in polys:
            polys_new.append(approximate_polygon(poly, tolerance=tol))
        polys = polys_new
        tol += 2.0
    return polys


def random_int(low, high):
    if low < high:
        return np.random.randint(low, high)
    else:
        return max(low, high)


def interpolate_points(ps, pe):
    xs, ys = ps
    xe, ye = pe
    points = []
    dx = xe - xs
    dy = ye - ys
    if dx != 0:
        scale = dy / dx
        if xe > xs:
            x_interpolated = list(range(ceil(xs), floor(xe) + 1))
        else:
            x_interpolated = list(range(floor(xs), ceil(xe) - 1, -1))
        for x in x_interpolated:
            y = ys + (x - xs) * scale
            points.append([x, y])
    if dy != 0:
        scale = dx / dy
        if ye > ys:
            y_interpolated = list(range(ceil(ys), floor(ye) + 1))
        else:
            y_interpolated = list(range(floor(ys), ceil(ye) - 1, -1))
        for y in y_interpolated:
            x = xs + (y - ys) * scale
            points.append([x, y])
    if xe > xs:
        points = sorted(points, key=lambda x: x[0])
    else:
        points = sorted(points, key=lambda x: -x[0])
    return points


def interpolate_polygon(polygon):
    points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    points_interpolated = []
    points_interpolated.append(points[0])
    for i in range(0, len(points) - 1):
        points_i = interpolate_points(points[i], points[i + 1])
        points_interpolated += points_i
        points_interpolated.append(points[i + 1])
    points_interpolated = prune_points(points_interpolated)
    polygon_interpolated = np.array(points_interpolated)
    return list(polygon_interpolated.flatten())


def prune_points(points, th=0.1):
    points_pruned = [points[0]]
    for i in range(1, len(points)):
        x1, y1 = points_pruned[-1]
        x2, y2 = points[i]
        dist = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if dist > th:
            points_pruned.append(points[i])
    return points_pruned


def interpolate_polygons(polygons):
    polygons_i = []
    for polygon in polygons:
        polygons_i.append(interpolate_polygon(polygon))
    return polygons_i


def sample_polygon(polygon, sample_rate=0.5):
    points = np.array(polygon).reshape(int(len(polygon) / 2), 2)
    k = int(len(points) * sample_rate)
    index = sorted(sample(list(range(len(points))), k))
    points_sampled = points[index]
    return list(np.array(points_sampled).flatten())


def sample_polygons(polygons, max_length=400.0):
    n = check_length(polygons)
    k = max_length / n
    polygons_s = []
    for polygon in polygons:
        polygons_s.append(sample_polygon(polygon, k))
    return polygons_s


def polygons_to_string(polygons):
    pts_strings = []
    for polygon in polygons:
        pts_string = ','.join([str(num) for num in polygon])
        pts_string += " "  # separator
        pts_strings.append(pts_string)
    pts_strings = "".join(pts_strings)
    return pts_strings

