// Splits a page image into text regions, off the main thread.
//
// 1. Bina's own pixel-only layout (/bina/layout.mjs) finds columns and
//    paragraph blocks.
// 2. Each block is trimmed to its content (white margins removed) and tall
//    blocks are cut further, always at blank rows between text lines, so that
//    no block has more than MAX_PATCHES 16x16 image patches. The model's cost
//    grows much faster than linearly with the patch count (measured: ~1,300
//    patches ≈ 5 s, ~4,000 patches > 50 s, a full A4 page minutes), so several
//    small blocks are far faster than one big one.
import {partitionPNG} from '/bina/layout.mjs';

const MAX_PATCHES = 1400;
const MODEL_MAX_W = 1664, MODEL_MAX_H = 2048;
const PAD = 12;
const INK = 170; // luminance below this counts as ink

function inkProfile(rgb, w, h) {
  const rows = new Uint32Array(h), cols = new Uint32Array(w);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const k = (y * w + x) * 3;
      if (0.299 * rgb[k] + 0.587 * rgb[k + 1] + 0.114 * rgb[k + 2] < INK) { rows[y]++; cols[x]++; }
    }
  }
  return {rows, cols};
}

function bounds(counts, min) {
  let a = 0, b = counts.length - 1;
  while (a <= b && counts[a] < min) a++;
  while (b >= a && counts[b] < min) b--;
  return a > b ? null : [a, b + 1];
}

// Tallest block (in source pixels) of width w that stays within MAX_PATCHES,
// following the model's preprocessing (shrink to fit 1664 wide, pad width to 64).
function maxHeightFor(w) {
  const scale = Math.min(1, MODEL_MAX_W / w);
  const cols = Math.ceil((w * scale) / 64) * 4;
  const rows = Math.max(4, Math.floor(MAX_PATCHES / cols));
  return Math.min(MODEL_MAX_H, Math.floor((rows * 16) / scale));
}

// Number of 16x16 patches the model will see for a w x h block (used for
// progress and time estimates).
function patchCount(w, h) {
  const scale = Math.min(1, MODEL_MAX_W / w, MODEL_MAX_H / h);
  return Math.ceil((w * scale) / 64) * 4 * Math.ceil((h * scale) / 16);
}

// Cut [top, bottom) into pieces no taller than maxH, at blank rows.
function cuts(rowInk, top, bottom, rowMin, maxH) {
  const out = [top];
  let start = top;
  while (bottom - start > maxH) {
    const lo = start + Math.floor(maxH * 0.45), hi = start + maxH;
    let best = -1, bestLen = 0, runStart = -1;
    for (let y = lo; y <= hi; y++) {
      const blank = y < hi && rowInk[y] < rowMin;
      if (blank && runStart < 0) runStart = y;
      if (!blank && runStart >= 0) {
        const len = y - runStart;
        // prefer longer gaps, and later ones on ties (fewer pieces)
        if (len >= bestLen) { bestLen = len; best = runStart + (len >> 1); }
        runStart = -1;
      }
    }
    const cut = best > 0 ? best : hi; // no blank row: hard cut (rare)
    out.push(cut);
    start = cut;
  }
  out.push(bottom);
  return out;
}

async function encode(rgb, w, x0, y0, cw, ch) {
  const rgba = new Uint8ClampedArray(cw * ch * 4);
  for (let y = 0; y < ch; y++) {
    for (let x = 0; x < cw; x++) {
      const s = ((y0 + y) * w + x0 + x) * 3, d = (y * cw + x) * 4;
      rgba[d] = rgb[s]; rgba[d + 1] = rgb[s + 1]; rgba[d + 2] = rgb[s + 2]; rgba[d + 3] = 255;
    }
  }
  const canvas = new OffscreenCanvas(cw, ch);
  canvas.getContext('2d').putImageData(new ImageData(rgba, cw, ch), 0, 0);
  return (await canvas.convertToBlob({type: 'image/png'})).arrayBuffer();
}

async function refine(region) {
  const [bx0, by0, bx1, by1] = region.bbox;
  const w = bx1 - bx0, h = by1 - by0, rgb = region.rgb;
  const {rows, cols} = inkProfile(rgb, w, h);
  const rowMin = Math.max(1, Math.floor(w * 0.002)), colMin = 1;
  const ys = bounds(rows, rowMin), xs = bounds(cols, colMin);
  if (!ys || !xs) return []; // blank block
  const x0 = Math.max(0, xs[0] - PAD), x1 = Math.min(w, xs[1] + PAD);
  const top = Math.max(0, ys[0] - PAD), bottom = Math.min(h, ys[1] + PAD);
  const ycuts = cuts(rows, top, bottom, rowMin, maxHeightFor(x1 - x0));
  const pieces = [];
  for (let i = 0; i < ycuts.length - 1; i++) {
    const y0 = ycuts[i], y1 = ycuts[i + 1];
    let ink = 0;
    for (let y = y0; y < y1; y++) ink += rows[y] >= rowMin ? 1 : 0;
    if (!ink) continue;
    pieces.push({
      bbox: [bx0 + x0, by0 + y0, bx0 + x1, by0 + y1],
      patches: patchCount(x1 - x0, y1 - y0),
      png: await encode(rgb, w, x0, y0, x1 - x0, y1 - y0),
    });
  }
  return pieces;
}

self.onmessage = async (event) => {
  const {id, png} = event.data;
  try {
    const plan = partitionPNG(png);
    const regions = [];
    for (const r of plan.regions) {
      for (const piece of await refine(r)) regions.push({index: regions.length, ...piece});
    }
    self.postMessage({id, regions, size: plan.original_size}, regions.map((r) => r.png));
  } catch (error) {
    self.postMessage({id, error: String(error && error.message || error)});
  }
};
