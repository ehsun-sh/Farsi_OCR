// Same-origin proxy for the Bina model files on Hugging Face.
//
// Hugging Face does not send CORS headers to some hosting domains (for example
// *.workers.dev), so the browser cannot download the model from there directly.
// This proxy fetches the files server-side and serves them from the site's own
// origin. It only serves the pinned model revision, so it is not an open proxy.

export const HF_REPO = 'Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4';
export const HF_REVISION = 'a4f59f1c659435721a65aa0d881c9a5cd7bfe7ee';
const ALLOWED = /^(model\/(assets|shared-models|fixtures)\/[\w.@-]+|ort\/ort-wasm-simd-threaded\.jsep\.wasm)$/;
const PASS_HEADERS = ['content-type', 'content-length', 'etag', 'last-modified', 'accept-ranges', 'content-range'];

// path: the part after "/hf/", e.g. "<revision>/model/assets/prompts.json"
export async function proxyHuggingFace(request, path) {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response('Method not allowed', {status: 405, headers: {allow: 'GET, HEAD'}});
  }
  const [revision, ...rest] = path.split('/');
  const file = rest.join('/');
  if (revision !== HF_REVISION || !ALLOWED.test(file)) return new Response('Not found', {status: 404});

  const headers = {};
  const range = request.headers.get('range');
  if (range) headers.range = range;
  const upstream = await fetch(`https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/${file}`, {
    method: request.method,
    headers,
    redirect: 'follow',
    cf: {cacheEverything: true, cacheTtl: 31536000},
  });
  const out = new Headers();
  for (const name of PASS_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  if (upstream.ok) out.set('cache-control', 'public, max-age=31536000, immutable');
  return new Response(upstream.body, {status: upstream.status, headers: out});
}
