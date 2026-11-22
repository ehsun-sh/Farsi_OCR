// The exact Bina release this app runs. Keep in sync with scripts/build.mjs.
export const BINA_REPO = 'Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4';
export const BINA_REVISION = 'a4f59f1c659435721a65aa0d881c9a5cd7bfe7ee';
// Model files are fetched either straight from Hugging Face or, when the site
// runs on Cloudflare with worker/hf-proxy.js, through the same-origin /hf/ proxy
// (Hugging Face refuses CORS for some hosting domains such as *.workers.dev).
export const HF_DIRECT_RESOLVE = `https://huggingface.co/${BINA_REPO}/resolve/${BINA_REVISION}/`;
export const HF_PROXY_RESOLVE = `/hf/${BINA_REVISION}/`;
export const BINA_CACHE_NAME = 'bina-0.4-koochik-a4f59f1c';
export const BINA_SOURCE_STEP = 18404;
export const BINA_CANDIDATE_SHA256 = '8b702a444a2ff70d478f4bd06875f19ec7fed2b6084d16013f1cf41c4ac7e469';
export const APPROX_DOWNLOAD_MB = 770;

export const PROMPTS = {
  text: '</s><s><predict_text_in_pic>',
  markdown: '</s><s><output_markdown><predict_text_in_pic>',
};
