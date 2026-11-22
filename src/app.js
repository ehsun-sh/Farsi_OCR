import {APPROX_DOWNLOAD_MB, BINA_CACHE_NAME, BINA_CANDIDATE_SHA256, BINA_SOURCE_STEP, PROMPTS} from './config.js';
import {BinaClient, LayoutClient} from './bina-client.js';
import {baseName, toDocx, toMarkdown, toText} from './exporters.js';

const $ = (id) => document.getElementById(id);
const fa = (n) => String(n).replace(/\d/g, (d) => '۰۱۲۳۴۵۶۷۸۹'[d]);
const mb = (bytes) => fa(Math.round(bytes / 1e6));
const MAX_IMAGE_PIXELS = 20_000_000;

const state = {
  client: null,        // BinaClient once the model is loaded
  loading: null,       // Promise while the model loads
  layout: null,
  files: [],           // {id, file, isPdf, pageCount, status}
  docs: [],            // results, see exporters.js
  running: false,
  stopping: false,
  gpuOk: false,
};

// ---------------------------------------------------------------------------
// WebGPU check

async function checkWebGPU() {
  const banner = $('gpu-banner');
  const show = (title, body, error = true) => {
    banner.className = 'banner' + (error ? ' error' : '');
    banner.innerHTML = `<h3>${title}</h3>${body}`;
    banner.hidden = false;
  };
  if (!window.isSecureContext) {
    show('این صفحه باید با HTTPS باز شود',
      '<p>WebGPU فقط روی HTTPS (یا localhost) کار می‌کند. نشانی را با <span dir="ltr">https://</span> باز کنید.</p>');
    return false;
  }
  if (!('gpu' in navigator)) {
    show('مرورگر شما از WebGPU پشتیبانی نمی‌کند',
      '<p>این برنامه مدل را روی کارت گرافیک خود شما اجرا می‌کند و برای این کار به WebGPU نیاز دارد.</p>' +
      '<p>لطفاً نسخهٔ جدید <b>Chrome</b> یا <b>Edge</b> را روی کامپیوتر (ویندوز، مک یا لینوکس) باز کنید. ' +
      'Firefox و Safari و بیشتر مرورگرهای موبایل فعلاً پشتیبانی کاملی ندارند.</p>' +
      '<p class="small">اجرای بدون کارت گرافیک (WASM) برای این مدل در دسترس نیست، چون نسخهٔ مرورگری بینا فقط برای WebGPU ساخته و آزموده شده است.</p>');
    return false;
  }
  let adapter = null;
  try { adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'}); } catch {}
  if (!adapter) {
    show('WebGPU روشن است ولی کارت گرافیکی پیدا نشد',
      '<p>ممکن است درایور کارت گرافیک قدیمی باشد یا شتاب‌دهی سخت‌افزاری مرورگر خاموش باشد.</p>' +
      '<p>در Chrome به <span dir="ltr">chrome://settings/system</span> بروید و «Use graphics acceleration when available» را روشن کنید، ' +
      'درایور گرافیک را به‌روز کنید و مرورگر را دوباره باز کنید. وضعیت را در <span dir="ltr">chrome://gpu</span> ببینید.</p>');
    return false;
  }
  if (/Android|iPhone|iPad|Mobile/i.test(navigator.userAgent)) {
    show('روی موبایل ممکن است کار نکند',
      `<p>مدل حدود ${fa(APPROX_DOWNLOAD_MB)} مگابایت است و به حافظهٔ گرافیکی زیادی نیاز دارد. اگر صفحه بسته شد یا خطا داد، از کامپیوتر استفاده کنید.</p>`, false);
  }
  return true;
}

async function modelIsCached() {
  try {
    if (!(await caches.has(BINA_CACHE_NAME))) return false;
    const keys = await (await caches.open(BINA_CACHE_NAME)).keys();
    return keys.some((r) => r.url.includes('decoder.int4.onnx.data'));
  } catch { return false; }
}

// ---------------------------------------------------------------------------
// Model loading

function setBar(el, fraction) {
  el.hidden = false;
  el.classList.toggle('indeterminate', fraction == null);
  el.querySelector('.bar').style.width = fraction == null ? '' : `${Math.min(100, Math.max(0, fraction * 100)).toFixed(1)}%`;
}

const STAGES = {
  'loading vision': 'آماده‌سازی بخش بینایی مدل…',
  'loading neck': 'آماده‌سازی بخش میانی مدل…',
  'loading prefill': 'آماده‌سازی رمزگشا…',
  'loading decode': 'آماده‌سازی رمزگشا…',
  'decoder cache control passed': 'آزمایش اولیهٔ مدل انجام شد…',
};

function loadModel() {
  if (state.client) return Promise.resolve(state.client);
  if (state.loading) return state.loading;
  const status = $('model-status'), bar = $('model-progress');
  status.classList.remove('error');
  $('load-model').disabled = true;
  $('clear-cache').disabled = true;
  $('model-state').textContent = 'در حال بارگذاری';
  status.textContent = 'در حال آماده‌سازی…';
  setBar(bar, null);

  const downloads = new Map();
  let fromCache = 0;
  const client = new BinaClient({
    onDownload: (m) => {
      downloads.set(m.url, m);
      if (m.cached) fromCache++;
      let loaded = 0, total = 0;
      for (const d of downloads.values()) { loaded += d.loaded; total += d.total || d.loaded; }
      const expected = Math.max(total, APPROX_DOWNLOAD_MB * 1e6);
      setBar(bar, loaded / expected);
      status.textContent = fromCache && fromCache === downloads.size
        ? `خواندن مدل از حافظهٔ مرورگر… ${mb(loaded)} مگابایت`
        : `دانلود مدل: ${mb(loaded)} از حدود ${fa(APPROX_DOWNLOAD_MB)} مگابایت`;
    },
    onProgress: (m) => {
      const text = STAGES[m.stage];
      if (text) { status.textContent = text; setBar(bar, null); }
    },
  });

  state.loading = client.initialize().then((ready) => {
    if (ready.source_step !== BINA_SOURCE_STEP || ready.candidate_sha256 !== BINA_CANDIDATE_SHA256) {
      throw new Error('نسخهٔ مدل با نسخهٔ مورد انتظار یکی نیست');
    }
    state.client = client;
    $('model-state').textContent = 'آماده';
    $('model-state').classList.add('ok');
    status.textContent = `مدل آماده است (${(ready.timing.totalMilliseconds / 1000).toFixed(1)} ثانیه). حالا فایل‌ها را اضافه کنید.`;
    bar.hidden = true;
    $('load-model').textContent = 'مدل بارگذاری شد';
    return client;
  }).catch((error) => {
    client.close();
    $('model-state').textContent = 'خطا';
    status.classList.add('error');
    status.textContent = 'بارگذاری مدل ناموفق بود: ' + friendlyError(error);
    bar.hidden = true;
    $('load-model').disabled = !state.gpuOk;
    throw error;
  }).finally(() => {
    state.loading = null;
    $('clear-cache').disabled = false;
    updateButtons();
  });
  return state.loading;
}

function friendlyError(error) {
  const msg = String(error && error.message || error);
  if (/device lost|Device was lost/i.test(msg)) return 'ارتباط با کارت گرافیک قطع شد (احتمالاً حافظهٔ گرافیکی کافی نیست). صفحه را دوباره باز کنید. — ' + msg;
  if (/QuotaExceeded|quota/i.test(msg)) return 'فضای کافی در مرورگر نیست. — ' + msg;
  if (/Failed to fetch|NetworkError|network/i.test(msg)) return 'اتصال اینترنت یا دسترسی به Hugging Face برقرار نشد. — ' + msg;
  if (/WebGPU/i.test(msg)) return 'WebGPU در دسترس نیست. — ' + msg;
  return msg;
}

// ---------------------------------------------------------------------------
// Files

let nextFileId = 1;
const isPdfFile = (f) => f.type === 'application/pdf' || /\.pdf$/i.test(f.name);
const isImageFile = (f) => f.type.startsWith('image/') || /\.(png|jpe?g|webp|bmp|gif)$/i.test(f.name);

function addFiles(list) {
  const rejected = [];
  for (const file of list) {
    if (isPdfFile(file) || isImageFile(file)) {
      state.files.push({id: nextFileId++, file, isPdf: isPdfFile(file), pageCount: null, status: ''});
    } else rejected.push(file.name);
  }
  if (rejected.length) $('run-status').textContent = 'این فایل‌ها پشتیبانی نمی‌شوند: ' + rejected.join('، ');
  renderFiles();
  for (const f of state.files) if (f.isPdf && f.pageCount == null) countPages(f);
}

async function countPages(entry) {
  try {
    const pdf = await openPdf(entry.file);
    entry.pageCount = pdf.numPages;
    closePdf(pdf);
  } catch {
    entry.pageCount = 0;
    entry.status = 'PDF خوانده نشد';
  }
  renderFiles();
}

function renderFiles() {
  const ul = $('file-list');
  ul.replaceChildren(...state.files.map((f) => {
    const li = document.createElement('li');
    const name = Object.assign(document.createElement('span'), {className: 'name', textContent: f.file.name, title: f.file.name});
    const meta = document.createElement('span');
    meta.className = 'meta';
    const kind = f.isPdf ? (f.pageCount ? `PDF، ${fa(f.pageCount)} صفحه` : 'PDF') : 'تصویر';
    meta.textContent = f.status ? `${kind} · ${f.status}` : kind;
    const remove = Object.assign(document.createElement('button'), {className: 'remove', textContent: '✕', title: 'حذف', disabled: state.running});
    remove.setAttribute('aria-label', 'حذف ' + f.file.name);
    remove.onclick = () => { state.files = state.files.filter((x) => x !== f); renderFiles(); };
    li.append(name, meta, remove);
    return li;
  }));
  updateButtons();
}

function updateButtons() {
  $('start').disabled = state.running || !state.gpuOk || !state.files.length;
  $('stop').disabled = !state.running;
  $('load-model').disabled = !state.gpuOk || !!state.client || !!state.loading;
  $('file-input').disabled = state.running;
}

// ---------------------------------------------------------------------------
// Turning inputs into PNG pages

let pdfjsPromise = null;
function pdfjs() {
  pdfjsPromise ??= import('./vendor/pdfjs/pdf.min.mjs').then((lib) => {
    lib.GlobalWorkerOptions.workerSrc = new URL('./vendor/pdfjs/pdf.worker.min.mjs', import.meta.url).href;
    return lib;
  });
  return pdfjsPromise;
}

async function openPdf(file) {
  const lib = await pdfjs();
  const base = new URL('./vendor/pdfjs/', import.meta.url).href;
  return lib.getDocument({
    data: new Uint8Array(await file.arrayBuffer()),
    cMapUrl: base + 'cmaps/', cMapPacked: true,
    standardFontDataUrl: base + 'standard_fonts/',
    wasmUrl: base + 'wasm/',
  }).promise;
}

function closePdf(pdf) {
  pdf.loadingTask?.destroy?.().catch?.(() => {});
}

function canvasToPng(canvas) {
  return new Promise((resolve, reject) => canvas.toBlob(
    (blob) => blob ? blob.arrayBuffer().then(resolve, reject) : reject(new Error('Could not encode PNG')), 'image/png'));
}

function whiteCanvas(width, height) {
  const canvas = document.createElement('canvas');
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d', {alpha: false});
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, width, height);
  return {canvas, ctx};
}

async function imageToPng(file) {
  const bitmap = await createImageBitmap(file, {imageOrientation: 'from-image'});
  let {width, height} = bitmap;
  const scale = Math.min(1, Math.sqrt(MAX_IMAGE_PIXELS / (width * height)));
  width = Math.max(1, Math.round(width * scale));
  height = Math.max(1, Math.round(height * scale));
  const {canvas, ctx} = whiteCanvas(width, height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();
  return canvasToPng(canvas);
}

async function pdfPageToPng(pdf, number, targetWidth) {
  const page = await pdf.getPage(number);
  const unit = page.getViewport({scale: 1});
  let scale = Math.min(4, targetWidth / unit.width);
  if (unit.height * scale > 4000) scale = 4000 / unit.height; // very tall pages
  const viewport = page.getViewport({scale});
  const {canvas, ctx} = whiteCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height));
  await page.render({canvas, canvasContext: ctx, viewport, background: '#ffffff'}).promise;
  page.cleanup();
  return canvasToPng(canvas);
}

function parsePages(spec, count) {
  spec = spec.trim().replace(/[۰-۹]/g, (d) => '۰۱۲۳۴۵۶۷۸۹'.indexOf(d)).replace(/[،,]/g, ',');
  if (!spec) return Array.from({length: count}, (_, i) => i + 1);
  const set = new Set();
  for (const part of spec.split(',')) {
    const m = /^\s*(\d+)\s*(?:[-–]\s*(\d+)?\s*)?$/.exec(part);
    if (!m) throw new Error(`بازهٔ صفحه نامعتبر است: «${part.trim()}»`);
    const a = +m[1], b = m[2] ? +m[2] : (part.includes('-') || part.includes('–') ? count : a);
    for (let p = Math.max(1, a); p <= Math.min(count, b); p++) set.add(p);
  }
  if (!set.size) throw new Error('هیچ صفحه‌ای در این بازه نیست');
  return [...set].sort((x, y) => x - y);
}

// ---------------------------------------------------------------------------
// Running OCR

async function run() {
  if (state.running) return;
  state.running = true;
  state.stopping = false;
  renderFiles();
  const status = $('run-status'), bar = $('run-progress');
  status.classList.remove('error');
  const mode = $('mode').value;
  const prompt = PROMPTS[mode];
  const pdfWidth = +$('pdf-width').value;
  const started = performance.now();

  try {
    const client = await loadModel();
    state.layout ??= new LayoutClient();

    // Plan the work first so progress is meaningful.
    const jobs = [];
    for (const entry of state.files) {
      if (entry.isPdf) {
        const pdf = await openPdf(entry.file);
        const pages = parsePages($('pages').value, pdf.numPages);
        jobs.push({entry, pdf, pages});
      } else jobs.push({entry, pdf: null, pages: [1]});
    }
    const totalPages = jobs.reduce((n, j) => n + j.pages.length, 0);
    let donePages = 0;
    $('results-card').hidden = false;

    for (const job of jobs) {
      const doc = {name: job.entry.file.name, isPdf: !!job.pdf, mode, pages: []};
      state.docs.push(doc);
      const docView = renderDoc(doc);
      job.entry.status = 'در حال پردازش';
      renderFiles();

      for (const number of job.pages) {
        if (state.stopping) throw new DOMException('Stopped', 'AbortError');
        const page = {number, text: ''};
        doc.pages.push(page);
        const view = addPageView(docView, doc, page);
        view.wrap.classList.add('working');
        const where = `${job.entry.file.name}${job.pdf ? ` — صفحهٔ ${fa(number)}` : ''}`;
        status.textContent = `${where}: آماده‌سازی تصویر…`;
        setBar(bar, donePages / totalPages);

        const png = job.pdf ? await pdfPageToPng(job.pdf, number, pdfWidth) : await imageToPng(job.entry.file);
        const {regions} = await state.layout.partition(png);
        const texts = [];
        const joiner = mode === 'markdown' ? '\n\n' : '\n';
        for (const region of regions) {
          if (state.stopping) throw new DOMException('Stopped', 'AbortError');
          const partLabel = regions.length > 1 ? ` (بخش ${fa(region.index + 1)} از ${fa(regions.length)})` : '';
          client.onProgress = (m) => {
            const tokens = m.tokens ? ` — ${fa(m.tokens)} توکن` : '';
            status.textContent = `${where}${partLabel}: در حال خواندن متن${tokens}`;
            if (m.text !== undefined) view.textarea.value = [...texts, m.text].filter(Boolean).join(joiner);
            setBar(bar, (donePages + (region.index + (m.stage === 'decode' ? .5 : .2)) / regions.length) / totalPages);
          };
          const result = await client.transcribe(region.png, prompt);
          texts.push(result.selected.ocr_prediction);
          page.text = texts.filter(Boolean).join(joiner);
          view.textarea.value = page.text;
        }
        view.wrap.classList.remove('working');
        donePages++;
      }
      job.entry.status = 'انجام شد';
      if (job.pdf) closePdf(job.pdf);
      renderFiles();
    }
    setBar(bar, 1);
    const seconds = (performance.now() - started) / 1000;
    status.textContent = `تمام شد: ${fa(totalPages)} صفحه در ${fa(Math.round(seconds))} ثانیه. فایل خروجی را از پایین دانلود کنید.`;
    state.files = [];
  } catch (error) {
    if (error.name === 'AbortError') status.textContent = 'متوقف شد. متن‌های خوانده‌شده تا این لحظه نگه داشته شده‌اند.';
    else {
      console.error(error);
      status.classList.add('error');
      status.textContent = 'خطا: ' + friendlyError(error);
    }
    for (const f of state.files) if (f.status === 'در حال پردازش') f.status = '';
  } finally {
    state.running = false;
    bar.hidden = true;
    document.querySelectorAll('.page.working').forEach((el) => el.classList.remove('working'));
    pruneEmptyDocs();
    renderFiles();
  }
}

// ---------------------------------------------------------------------------
// Results view

function renderDoc(doc) {
  const el = document.createElement('div');
  el.className = 'doc';
  el.append(Object.assign(document.createElement('h3'), {textContent: doc.name}));
  $('results').append(el);
  doc.el = el;
  return el;
}

function addPageView(docEl, doc, page) {
  const wrap = document.createElement('div');
  wrap.className = 'page';
  if (doc.isPdf) wrap.append(Object.assign(document.createElement('div'), {className: 'page-label', textContent: `صفحهٔ ${fa(page.number)}`}));
  const textarea = document.createElement('textarea');
  textarea.dir = 'auto';
  textarea.spellcheck = false;
  textarea.setAttribute('aria-label', `متن ${doc.name}${doc.isPdf ? ' صفحهٔ ' + page.number : ''}`);
  textarea.addEventListener('input', () => { page.text = textarea.value; });
  wrap.append(textarea);
  docEl.append(wrap);
  return {wrap, textarea};
}

function pruneEmptyDocs() {
  state.docs = state.docs.filter((d) => {
    d.pages = d.pages.filter((p) => p.text.trim());
    if (!d.pages.length) { d.el?.remove(); return false; }
    return true;
  });
  // remove empty page views left by a stop
  document.querySelectorAll('.page textarea').forEach((t) => { if (!t.value.trim()) t.closest('.page').remove(); });
  $('results-card').hidden = !state.docs.length;
}

function download(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), {href: url, download: name});
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

async function exportAs(kind) {
  const docs = state.docs.filter((d) => d.pages.length);
  if (!docs.length) return;
  const name = baseName(docs);
  if (kind === 'txt') download(new Blob(['﻿' + toText(docs)], {type: 'text/plain;charset=utf-8'}), name + '.txt');
  if (kind === 'md') download(new Blob([toMarkdown(docs)], {type: 'text/markdown;charset=utf-8'}), name + '.md');
  if (kind === 'docx') {
    try { download(await toDocx(docs), name + '.docx'); }
    catch (error) { $('run-status').textContent = 'ساخت فایل Word ناموفق بود: ' + error.message; }
  }
}

// ---------------------------------------------------------------------------
// Wiring

async function init() {
  const drop = $('drop'), input = $('file-input');
  input.onchange = () => { addFiles(input.files); input.value = ''; };
  drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  for (const t of ['dragenter', 'dragover']) drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); });
  for (const t of ['dragleave', 'drop']) drop.addEventListener(t, () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => { e.preventDefault(); if (!state.running) addFiles(e.dataTransfer.files); });
  document.addEventListener('paste', (e) => {
    if (state.running || e.target.closest?.('textarea, input')) return;
    const files = [...(e.clipboardData?.files || [])].map((f, i) =>
      f.name && f.name !== 'image.png' ? f : new File([f], `pasted-${Date.now()}-${i + 1}.png`, {type: f.type}));
    if (files.length) addFiles(files);
  });

  $('load-model').onclick = () => loadModel().catch(() => {});
  $('start').onclick = run;
  $('stop').onclick = () => { state.stopping = true; state.client?.cancel(); $('run-status').textContent = 'در حال توقف…'; };
  $('clear-results').onclick = () => { state.docs = []; $('results').replaceChildren(); $('results-card').hidden = true; };
  $('copy-all').onclick = async () => {
    try { await navigator.clipboard.writeText(toText(state.docs)); $('copy-all').textContent = 'کپی شد ✓'; }
    catch { $('copy-all').textContent = 'کپی نشد'; }
    setTimeout(() => { $('copy-all').textContent = 'کپی همهٔ متن'; }, 2000);
  };
  document.querySelectorAll('[data-export]').forEach((b) => { b.onclick = () => exportAs(b.dataset.export); });
  $('clear-cache').onclick = async () => {
    if (state.running || state.loading) return;
    state.client?.close();
    state.client = null;
    await caches.delete(BINA_CACHE_NAME).catch(() => {});
    $('model-state').textContent = 'آماده نشده';
    $('model-state').classList.remove('ok');
    $('load-model').textContent = 'بارگذاری مدل';
    $('model-status').textContent = 'مدل از حافظهٔ مرورگر پاک شد.';
    updateButtons();
  };

  state.gpuOk = await checkWebGPU();
  if (!state.gpuOk) {
    $('model-note').textContent = 'تا وقتی WebGPU در دسترس نباشد، مدل اجرا نمی‌شود.';
  } else if (await modelIsCached()) {
    $('model-note').textContent = 'مدل قبلاً در این مرورگر ذخیره شده و بدون دانلود دوباره بالا می‌آید.';
  }
  if (navigator.storage?.persist) navigator.storage.persist().catch(() => {});
  updateButtons();
}

init();
