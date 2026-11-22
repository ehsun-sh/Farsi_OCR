"""Image preprocessing for Bina 0.4 Koochik, in numpy.

A line-by-line port of preprocess.js from Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4
(positional geometry mirrors NVIDIA's hf_prefix_geometry.py, OpenMDW-1.1; the RGB
resize follows Pillow's separable bilinear/bicubic filter contract). Float32
rounding happens at the same steps as the JavaScript `Math.fround` calls, so the
model sees the same inputs as in the browser.
"""
import math

import numpy as np

F = np.float32
CHANNELS = 1152
SOURCE = 128
MAX_H, MAX_W = 2048, 1664


def round_even(x):
    a = math.floor(x)
    r = x - a
    return a + (a % 2) if r == 0.5 else math.floor(x + 0.5)


def _coefficients(n_in, n_out, cubic=False, integer=False):
    """Returns (starts[n_out], weights[n_out, taps]); padded taps have weight 0."""
    scale = n_in / n_out
    filter_scale = max(1.0, scale)
    support = filter_scale * (2 if cubic else 1)
    starts, rows = [], []
    for i in range(n_out):
        center = (i + 0.5) * scale
        start = max(0, math.trunc(center - support + 0.5))
        end = min(n_in, math.trunc(center + support + 0.5))
        ws = []
        for j in range(start, end):
            x = abs((j - center + 0.5) / filter_scale)
            if cubic:
                w = (1.5 * x - 2.5) * x * x + 1 if x < 1 else (((-0.5 * x + 2.5) * x - 4) * x + 2) if x < 2 else 0.0
            else:
                w = max(0.0, 1 - x)
            ws.append(w)
        total = sum(ws)
        if integer:
            ws = [math.trunc(w / total * 4194304 + (-0.5 if w < 0 else 0.5)) for w in ws]
        else:
            ws = [F(w / total) for w in ws]
        starts.append(start)
        rows.append(ws)
    taps = max(len(r) for r in rows)
    weights = np.zeros((n_out, taps), np.int64 if integer else np.float32)
    for i, r in enumerate(rows):
        weights[i, :len(r)] = r
    return np.array(starts, np.int64), weights


def _resample_axis(data, n_out, axis, cubic):
    """Integer separable resample of uint8 `data` along `axis` (0 = rows, 1 = cols)."""
    n_in = data.shape[axis]
    starts, weights = _coefficients(n_in, n_out, cubic, integer=True)
    src = np.moveaxis(data, axis, 0).astype(np.int64)
    acc = np.full((n_out,) + src.shape[1:], 2097152, np.int64)
    for k in range(weights.shape[1]):
        idx = np.minimum(starts + k, n_in - 1)
        w = weights[:, k].reshape((n_out,) + (1,) * (src.ndim - 1))
        acc += src[idx] * w
    out = np.clip(acc // 4194304, 0, 255).astype(np.uint8)
    return np.moveaxis(out, 0, axis)


def resize_rgb(rgb, width, height, cubic=False):
    """rgb: uint8 array (h, w, 3)."""
    h, w = rgb.shape[:2]
    if (w, h) == (width, height):
        return rgb
    out = rgb if width == w else _resample_axis(rgb, width, 1, cubic)
    return out if height == h else _resample_axis(out, height, 0, cubic)


def primary_image(rgb):
    h, w = rgb.shape[:2]
    if h >= 64:
        return rgb
    scale = min(4, max(2, 64 / h), 1664 / w)
    return resize_rgb(rgb, round_even(w * scale), round_even(h * scale), True) if scale > 1 else rgb


def retry_image(original, current):
    """Higher-resolution retry input, or None when it would not be larger."""
    h, w = original.shape[:2]
    scale = min(4, 2048 / h, 1664 / w)
    width, height = round_even(w * scale), round_even(h * scale)
    if scale > 1 and width * height > current.shape[0] * current.shape[1]:
        return resize_rgb(original, width, height, True)
    return None


def patch_grid(width, height):
    """(rows, cols) of 16x16 patches the model sees for an image of this size."""
    ratio = width / height
    if height > MAX_H:
        height = MAX_H
        width = math.trunc(height * ratio)
    if width > MAX_W:
        width = MAX_W
        height = math.trunc(width / ratio)
    return math.ceil(height / 16), math.ceil(width / 64) * 4


def process_pixels(rgb, config):
    height, width = rgb.shape[:2]
    ratio = width / height
    if height > MAX_H:
        height = MAX_H
        width = math.trunc(height * ratio)
    if width > MAX_W:
        width = MAX_W
        height = math.trunc(width / ratio)
    image = resize_rgb(rgb, width, height, False)
    padded_h, padded_w = math.ceil(height / 16) * 16, math.ceil(width / 64) * 64
    rows, cols = padded_h // 16, padded_w // 16
    canvas = np.full((padded_h, padded_w, 3), 255, np.uint8)
    canvas[:height, :width] = image
    mean = np.array(config['image_mean'], np.float64).astype(F)
    std = np.array(config['image_std'], np.float64).astype(F)
    v = (np.arange(256, dtype=np.float64) / 255).astype(F)
    if config.get('do_normalize', True):
        lookup = np.stack([((v - mean[c]).astype(F) / std[c]).astype(F) for c in range(3)])
    else:
        lookup = np.stack([v] * 3)
    values = np.stack([lookup[c][canvas[:, :, c]] for c in range(3)], axis=-1)  # (H, W, 3)
    patches = values.reshape(rows, 16, cols, 16, 3).transpose(0, 2, 4, 1, 3).reshape(rows * cols, 768)
    return {'patches': np.ascontiguousarray(patches, dtype=F), 'image_size': (padded_h, padded_w),
            'content_size': (height, width), 'rows': rows, 'cols': cols}


def _interpolated_field(field, rows, cols):
    if min(rows, cols) == 1:
        out = np.zeros((rows, cols, CHANNELS), F)
        for y in range(rows):
            y0, y1 = math.floor(y * SOURCE / rows), math.ceil((y + 1) * SOURCE / rows)
            for x in range(cols):
                x0, x1 = math.floor(x * SOURCE / cols), math.ceil((x + 1) * SOURCE / cols)
                acc = np.zeros(CHANNELS, F)
                for yy in range(y0, y1):
                    for xx in range(x0, x1):
                        acc = (acc + field[yy, xx]).astype(F)
                out[y, x] = (acc / F((y1 - y0) * (x1 - x0))).astype(F)
        return out
    xs, xw = _coefficients(SOURCE, cols)
    ys, yw = _coefficients(SOURCE, rows)
    horizontal = np.zeros((SOURCE, cols, CHANNELS), F)
    for x in range(cols):
        for j in range(xw.shape[1]):
            if xw[x, j] == 0 and xs[x] + j >= SOURCE:
                continue
            horizontal[:, x] = (horizontal[:, x] + (field[:, min(xs[x] + j, SOURCE - 1)] * xw[x, j]).astype(F)).astype(F)
    out = np.zeros((rows, cols, CHANNELS), F)
    for y in range(rows):
        for j in range(yw.shape[1]):
            if yw[y, j] == 0 and ys[y] + j >= SOURCE:
                continue
            out[y] = (out[y] + (horizontal[min(ys[y] + j, SOURCE - 1)] * yw[y, j]).astype(F)).astype(F)
    return out


def _axis(i, content, size):
    start = i * 16
    extent = max(0, min(16, content - start))
    norm = F(min(F(start + F(0.5 * extent)), content) / content)
    grid = F(F(2 * norm) - F(1))
    return max(0.0, min(size - 1, float(F(F(F(grid + F(1)) * F(size)) - F(1))) * 0.5))


def positions(field, content_size, image_size):
    """field: float32 (128, 128, 1152). Returns float32 (rows*cols, 1152), BF16-rounded."""
    height, width = content_size
    rows, cols = image_size[0] // 16, image_size[1] // 16
    cr, cc = math.ceil(height / 16), math.ceil(width / 16)
    inter = _interpolated_field(field, cr, cc).reshape(cr * cc, CHANNELS)

    def setup(n, content, size, limit):
        pos = np.array([_axis(i, content, size) for i in range(n)])
        p0 = np.floor(pos).astype(np.int64)
        p1 = np.minimum(limit - 1, p0 + 1)
        d = (pos - p0).astype(F)
        return p0, p1, d

    y0, y1, dy = setup(rows, height, cr, cr)
    x0, x1, dx = setup(cols, width, cc, cc)
    one = F(1)
    wy0, wx0 = (one - dy).astype(F), (one - dx).astype(F)
    corners = [
        (y0[:, None] * cc + x0[None, :], (wy0[:, None] * wx0[None, :]).astype(F)),
        (y0[:, None] * cc + x1[None, :], (wy0[:, None] * dx[None, :]).astype(F)),
        (y1[:, None] * cc + x0[None, :], (dy[:, None] * wx0[None, :]).astype(F)),
        (y1[:, None] * cc + x1[None, :], (dy[:, None] * dx[None, :]).astype(F)),
    ]
    value = np.zeros((rows, cols, CHANNELS), F)
    for index, weight in corners:
        value = (value + (inter[index] * weight[..., None]).astype(F)).astype(F)
    bits = value.reshape(-1).view(np.uint32).astype(np.uint64)
    bits = ((bits + 32767 + ((bits >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    return bits.view(F).reshape(rows * cols, CHANNELS)


def load_field(path):
    field = np.load(path)
    return np.ascontiguousarray(field.astype(F).reshape(SOURCE, SOURCE, CHANNELS))
