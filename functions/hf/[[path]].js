// Cloudflare Pages Function: same proxy as the Worker, for Pages deployments.
import {proxyHuggingFace} from '../../worker/hf-proxy.js';

export const onRequest = ({request, params}) =>
  proxyHuggingFace(request, [].concat(params.path || []).join('/'));
