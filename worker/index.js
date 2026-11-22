// Cloudflare Worker entry: serves the static site from dist/ and proxies the
// model files under /hf/ (see hf-proxy.js).
import {proxyHuggingFace} from './hf-proxy.js';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith('/hf/')) return proxyHuggingFace(request, url.pathname.slice(4));
    return env.ASSETS.fetch(request);
  },
};
