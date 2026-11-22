// Splits a page image into text regions with Bina's own pixel-only layout code,
// off the main thread.
import {partitionPNG} from '/bina/layout.mjs';

self.onmessage = (event) => {
  const {id, png} = event.data;
  try {
    const plan = partitionPNG(png);
    const regions = plan.regions.map((r) => {
      const bytes = r.png.buffer.slice(r.png.byteOffset, r.png.byteOffset + r.png.byteLength);
      return {index: r.index, bbox: r.bbox, png: bytes};
    });
    self.postMessage({id, regions, size: plan.original_size}, regions.map((r) => r.png));
  } catch (error) {
    self.postMessage({id, error: String(error && error.message || error)});
  }
};
