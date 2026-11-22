// ---- Farsi OCR prelude (added at build time, not part of the original Bina worker) ----
// Wraps fetch() inside the inference worker so that:
//  * model files downloaded from Hugging Face are stored in Cache Storage and are
//    not downloaded again on the next visit;
//  * the page gets download progress messages ({kind: 'download', ...}).
{
  const CACHE_NAME = 'bina-0.4-koochik-a4f59f1c';
  const CACHEABLE = 'https://huggingface.co/Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4/resolve/a4f59f1c659435721a65aa0d881c9a5cd7bfe7ee/';
  const originalFetch = self.fetch.bind(self);
  const report = (url, loaded, total, cached, done) =>
    self.postMessage({kind: 'download', url, loaded, total, cached, done});

  self.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
    if (!url.startsWith(CACHEABLE) || (init && init.method && init.method !== 'GET')) return originalFetch(input, init);
    let cache = null;
    try {
      cache = await caches.open(CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        const size = +hit.headers.get('content-length') || 0;
        report(url, size, size, true, true);
        return hit;
      }
    } catch {
      cache = null; // Cache Storage unavailable (private mode etc.)
    }
    const response = await originalFetch(input, init);
    if (response.status !== 200 || !response.body) return response;
    const total = +response.headers.get('content-length') || 0;
    const headers = {'content-type': response.headers.get('content-type') || 'application/octet-stream'};
    if (total) headers['content-length'] = String(total);
    let forPage = response.body;
    if (cache) {
      const [a, b] = response.body.tee();
      forPage = a;
      cache.put(url, new Response(b, {headers})).catch(() => {});
    }
    let loaded = 0, last = 0;
    const counted = forPage.pipeThrough(new TransformStream({
      transform(chunk, controller) {
        loaded += chunk.byteLength;
        const now = performance.now();
        if (now - last > 200) { last = now; report(url, loaded, total, false, false); }
        controller.enqueue(chunk);
      },
      flush() { report(url, loaded, total || loaded, false, true); },
    }));
    return new Response(counted, {status: 200, statusText: 'OK', headers});
  };
}
// ---- end of prelude ----
