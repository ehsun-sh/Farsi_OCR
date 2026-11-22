// Build the static site into dist/.
//
// 1. Copies our own app (src/) into dist/.
// 2. Copies pdf.js and docx from node_modules into dist/vendor/.
// 3. Downloads the small browser runtime files of the Bina model from Hugging Face
//    (pinned to one revision, checked by SHA-256), patches the worker so that it
//    can be hosted on Cloudflare Pages, and writes them into dist/bina/.
//
// The model weights (~770 MB) are NOT copied: the visitor's browser downloads
// them straight from Hugging Face the first time and keeps them in Cache Storage.

import {createHash} from 'node:crypto';
import {cp, mkdir, readFile, rm, writeFile} from 'node:fs/promises';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const nm = join(root, 'node_modules');

// Keep in sync with src/config.js
const BINA_REPO = 'Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4';
const BINA_REVISION = 'a4f59f1c659435721a65aa0d881c9a5cd7bfe7ee';
const BINA_RESOLVE = `https://huggingface.co/${BINA_REPO}/resolve/${BINA_REVISION}/`;

// Files served from our own origin (module scripts must be same-origin and
// Hugging Face serves .js/.mjs as text/plain). Hashes pin the exact bytes.
const BINA_FILES = {
  'resident/inference-worker.js': 'd08507dbbe693b55f36ccc11ec227a8d5a6a1a35f78365bc9dafc7d50a356534',
  'custom-runtime/ort.jsep-bina.mjs': '6513d64ef40a4ddf3a27717895e5a11a01e71e444ac77f7a2f20d83db5c15ddc',
  'ort/ort-wasm-simd-threaded.jsep.mjs': '7838b4b2e77d6f1a739cd0de800e7f18a089328b365e6af4d416a88a94fd7fd3',
  'layout.mjs': '8527434a335f407573ba91a24cdf7ad6b1d524fff1f5e3517f97794ca3a056f3',
  // License and notices that must travel with any redistribution (OpenMDW-1.1).
  'LICENSE': '3375384ee1050a54d69c98c74d6499841166cd7db823ed90358f430df8c39856',
  'licenses/@huggingface-tokenizers-LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
  'licenses/fast-png-LICENSE': '47ec9747cf2c470a505fcb20f3ec2ff2e60ea4febaf2e1e7041a24fc369fac3c',
  'custom-runtime/ONNXRUNTIME-LICENSE': '2f07c72751aed99790b8a4869cf2311df85a860b22ded05fa22803587a48922c',
  'custom-runtime/build-receipt.json': '1346f45327810a22cda6d7232f2a3a39a5c007875262a33bc032237b3e9d4c8c',
};

// The ONNX Runtime .wasm is 29.5 MB, above Cloudflare's 25 MiB per-file asset
// limit, so the worker loads it from Hugging Face (directly or through the /hf/
// proxy; worker-prelude.js sets self.__BINA_RESOLVE) instead of from /ort/.
const WORKER_PATCHES = [
  [
    'ort = await import("/custom-runtime/ort.jsep-bina.mjs");\n  ort.env.wasm.wasmPaths = "/ort/";\n  ort.env.wasm.numThreads = 2;',
    'ort = await import("/bina/custom-runtime/ort.jsep-bina.mjs");\n' +
      '  ort.env.wasm.wasmPaths = { mjs: new URL("/bina/ort/ort-wasm-simd-threaded.jsep.mjs", self.location.href).href, wasm: self.__BINA_RESOLVE + "ort/ort-wasm-simd-threaded.jsep.wasm" };\n' +
      '  ort.env.wasm.numThreads = self.crossOriginIsolated ? 2 : 1;',
  ],
];

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');

async function download(path, expected) {
  for (let attempt = 1; ; attempt++) {
    try {
      const response = await fetch(BINA_RESOLVE + path);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const buf = Buffer.from(await response.arrayBuffer());
      const got = sha256(buf);
      if (got !== expected) throw new Error(`SHA-256 mismatch for ${path}: ${got}`);
      return buf;
    } catch (error) {
      if (attempt >= 4) throw new Error(`Could not download ${path} from Hugging Face: ${error.message}`);
      await new Promise((r) => setTimeout(r, 1000 * 2 ** attempt));
    }
  }
}

async function put(rel, data) {
  const file = join(dist, rel);
  await mkdir(dirname(file), {recursive: true});
  await writeFile(file, data);
}

await rm(dist, {recursive: true, force: true});
await cp(join(root, 'src'), dist, {recursive: true});

// --- vendor libraries -------------------------------------------------------
const pdfjs = join(nm, 'pdfjs-dist');
await cp(join(pdfjs, 'build/pdf.min.mjs'), join(dist, 'vendor/pdfjs/pdf.min.mjs'));
await cp(join(pdfjs, 'build/pdf.worker.min.mjs'), join(dist, 'vendor/pdfjs/pdf.worker.min.mjs'));
await cp(join(pdfjs, 'cmaps'), join(dist, 'vendor/pdfjs/cmaps'), {recursive: true});
await cp(join(pdfjs, 'standard_fonts'), join(dist, 'vendor/pdfjs/standard_fonts'), {recursive: true});
await cp(join(pdfjs, 'wasm'), join(dist, 'vendor/pdfjs/wasm'), {recursive: true});
await cp(join(pdfjs, 'LICENSE'), join(dist, 'vendor/pdfjs/LICENSE'));
await cp(join(nm, 'docx/dist/index.iife.js'), join(dist, 'vendor/docx/docx.iife.js'));
await cp(join(nm, 'docx/LICENSE'), join(dist, 'vendor/docx/LICENSE'));

// --- Bina runtime -----------------------------------------------------------
if (process.env.BINA_SKIP_DOWNLOAD) {
  console.warn('BINA_SKIP_DOWNLOAD is set: dist/bina/ was not created (UI-only build, OCR will not work).');
  process.exit(0);
}
for (const [path, hash] of Object.entries(BINA_FILES)) {
  let buf = await download(path, hash);
  if (path === 'resident/inference-worker.js') {
    let text = buf.toString('utf8');
    for (const [from, to] of WORKER_PATCHES) {
      const count = text.split(from).length - 1;
      if (count !== 1) throw new Error(`Worker patch target found ${count} times (expected 1)`);
      text = text.replace(from, () => to);
    }
    // Prepend a fetch wrapper: caches model files in Cache Storage and reports
    // download progress to the page.
    const prelude = await readFile(join(root, 'scripts/worker-prelude.js'), 'utf8');
    buf = Buffer.from(prelude + '\n' + text, 'utf8');
  }
  await put('bina/' + path, buf);
  console.log('bina/' + path);
}

console.log('Built into dist/');
