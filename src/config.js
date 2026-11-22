// The exact Bina release this app runs. Keep in sync with scripts/build.mjs.
export const BINA_REPO = 'Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4';
export const BINA_REVISION = 'a4f59f1c659435721a65aa0d881c9a5cd7bfe7ee';
export const BINA_MODEL_BASE = `https://huggingface.co/${BINA_REPO}/resolve/${BINA_REVISION}/model`;
export const BINA_CACHE_NAME = 'bina-0.4-koochik-a4f59f1c';
export const BINA_SOURCE_STEP = 18404;
export const BINA_CANDIDATE_SHA256 = '8b702a444a2ff70d478f4bd06875f19ec7fed2b6084d16013f1cf41c4ac7e469';
export const APPROX_DOWNLOAD_MB = 770;

export const PROMPTS = {
  text: '</s><s><predict_text_in_pic>',
  markdown: '</s><s><output_markdown><predict_text_in_pic>',
};
