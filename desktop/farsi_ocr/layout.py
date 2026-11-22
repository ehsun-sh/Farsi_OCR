"""Splits a page into text blocks small enough for the model to read quickly.

1. `page_regions` is Bina's own pixel-only layout (layout_regions.py from
   Reza2kn/Bina-0.4-Koochik-ONNX-INT4, OpenMDW-1.1): columns and paragraph blocks.
2. `refine` trims each block to its content and cuts tall blocks at blank rows
   so that no block has more than MAX_PATCHES 16x16 patches. The model's cost
   grows much faster than linearly with the patch count, so several small
   blocks are far faster than one big one. Same logic as the web app.
"""
import math

import numpy as np

from .preprocess import patch_grid

MAX_PATCHES = 1400
PAD = 12
INK = 170
LUMA = np.array([.299, .587, .114])


def _runs(mask):
    a = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    return list(zip(a[::2].tolist(), a[1::2].tolist()))


def page_regions(rgb):
    height, width = rgb.shape[:2]
    cuts = {0, width}
    margin = max(1, height // 50)
    med = np.median(rgb[margin:height - margin], axis=0)
    lum = med @ LUMA
    for start, end in _runs(lum < 240):
        if end - start < width * .08:
            continue
        if end > width * .98 and start > width * .45:
            cuts.add(start)
        elif start < width * .02 and end < width * .55:
            cuts.add(end)
    columns = list(zip(sorted(cuts)[:-1], sorted(cuts)[1:]))
    largest = max(b - a for a, b in columns)
    columns.sort(key=lambda ab: (-(ab[1] - ab[0]), -ab[0]) if largest >= width * .65 else (-ab[0], 0))
    result = []
    for x0, x1 in columns:
        ys = [0, height]
        if height >= 512:
            inset = max(2, (x1 - x0) // 50)
            part = rgb[:, x0 + inset:x1 - inset]
            ink = (part @ LUMA) < 110
            occupied = ink.sum(1) > max(2, int((x1 - x0) * .002))
            active = _runs(occupied)
            sizes = [b - a for a, b in active if b - a >= 3]
            threshold = max(48, 2.0 * (np.median(sizes) if sizes else 12), height * .025)
            ys = [0] + [(a[1] + b[0]) // 2 for a, b in zip(active, active[1:]) if b[0] - a[1] >= threshold] + [height]
            changed = True
            while changed and len(ys) > 2:
                changed = False
                for i, (a, b) in enumerate(zip(ys, ys[1:])):
                    if b - a < 128 or int(occupied[a:b].sum()) < 8:
                        ys.pop(i if i else 1)
                        changed = True
                        break
        for y0, y1 in zip(ys, ys[1:]):
            result.append((x0, y0, x1, y1))
    return result


def _bounds(counts, minimum):
    idx = np.flatnonzero(counts >= minimum)
    return None if idx.size == 0 else (int(idx[0]), int(idx[-1]) + 1)


def max_height_for(w):
    scale = min(1, 1664 / w)
    cols = math.ceil((w * scale) / 64) * 4
    rows = max(4, MAX_PATCHES // cols)
    return min(2048, math.floor((rows * 16) / scale))


def patch_count(w, h):
    rows, cols = patch_grid(w, h)
    return rows * cols


def _cuts(row_ink, top, bottom, row_min, max_h):
    out = [top]
    start = top
    while bottom - start > max_h:
        lo, hi = start + math.floor(max_h * 0.45), start + max_h
        best, best_len, run_start = -1, 0, -1
        for y in range(lo, hi + 1):
            blank = y < hi and row_ink[y] < row_min
            if blank and run_start < 0:
                run_start = y
            if not blank and run_start >= 0:
                length = y - run_start
                if length >= best_len:
                    best_len, best = length, run_start + (length >> 1)
                run_start = -1
        cut = best if best > 0 else hi
        out.append(cut)
        start = cut
    out.append(bottom)
    return out


def refine(rgb, box):
    bx0, by0, bx1, by1 = box
    block = rgb[by0:by1, bx0:bx1]
    h, w = block.shape[:2]
    ink = (block.astype(np.float32) @ LUMA.astype(np.float32)) < INK
    rows, cols = ink.sum(1), ink.sum(0)
    row_min = max(1, math.floor(w * 0.002))
    ys, xs = _bounds(rows, row_min), _bounds(cols, 1)
    if not ys or not xs:
        return []
    x0, x1 = max(0, xs[0] - PAD), min(w, xs[1] + PAD)
    top, bottom = max(0, ys[0] - PAD), min(h, ys[1] + PAD)
    ycuts = _cuts(rows, top, bottom, row_min, max_height_for(x1 - x0))
    pieces = []
    for y0, y1 in zip(ycuts, ycuts[1:]):
        if not (rows[y0:y1] >= row_min).any():
            continue
        pieces.append({'bbox': (bx0 + x0, by0 + y0, bx0 + x1, by0 + y1),
                       'patches': patch_count(x1 - x0, y1 - y0)})
    return pieces


def blocks(rgb):
    """Reading-order list of {'bbox': (x0, y0, x1, y1), 'patches': n}."""
    out = []
    for box in page_regions(rgb):
        out.extend(refine(rgb, box))
    return out
