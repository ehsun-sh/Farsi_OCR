// Talks to Bina's resident inference worker (adapted from the client shipped
// with Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4, OpenMDW-1.1).
import {BINA_MODEL_BASE} from './config.js';

export class BinaClient {
  constructor({onProgress = () => {}, onDownload = () => {}} = {}) {
    this.worker = new Worker('/bina/resident/inference-worker.js', {type: 'module'});
    this.pending = null;
    this.nextId = 0;
    this.onProgress = onProgress;
    this.onDownload = onDownload;
    this.closed = false;
    this.worker.onmessage = (event) => {
      const m = event.data;
      if (m.kind === 'download') { this.onDownload(m); return; }
      if (!this.pending || m.requestId !== this.pending.id) return;
      if (m.kind === 'progress') { this.onProgress(m); return; }
      const p = this.pending;
      if (['ready', 'result', 'unloaded'].includes(m.kind)) { this.pending = null; p.resolve(m); }
      else if (['error', 'cancelled'].includes(m.kind)) {
        this.pending = null;
        const e = new Error(m.error);
        e.name = m.kind === 'cancelled' ? 'AbortError' : 'Error';
        p.reject(e);
      }
    };
    this.worker.onerror = (event) => {
      const e = new Error(event.message || 'Worker failed to start');
      if (this.pending) { this.pending.reject(e); this.pending = null; }
      else this.startupError = e;
    };
  }

  request(message, transfer = []) {
    if (this.closed) return Promise.reject(new Error('Model unloaded'));
    if (this.startupError) return Promise.reject(this.startupError);
    if (this.pending) return Promise.reject(new Error('Another request is running'));
    return new Promise((resolve, reject) => {
      const id = ++this.nextId;
      this.pending = {id, resolve, reject};
      this.worker.postMessage({...message, requestId: id}, transfer);
    });
  }

  initialize() {
    return this.request({type: 'initialize', provider: 'webgpu', base: BINA_MODEL_BASE});
  }

  async transcribe(pngBuffer, prompt) {
    const reply = await this.request(
      {image: pngBuffer, provider: 'webgpu', prompt, base: BINA_MODEL_BASE, repetitionGuard: false},
      [pngBuffer],
    );
    return reply.result;
  }

  cancel() { this.worker.postMessage({type: 'cancel'}); }

  close() {
    this.closed = true;
    if (this.pending) { this.pending.reject(new Error('Model worker closed')); this.pending = null; }
    this.worker.terminate();
  }
}

export class LayoutClient {
  constructor() {
    this.worker = new Worker(new URL('./layout-worker.js', import.meta.url), {type: 'module'});
    this.waiting = new Map();
    this.nextId = 0;
    this.worker.onmessage = (event) => {
      const {id, error, regions, size} = event.data;
      const p = this.waiting.get(id);
      if (!p) return;
      this.waiting.delete(id);
      error ? p.reject(new Error(error)) : p.resolve({regions, size});
    };
    this.worker.onerror = (event) => {
      for (const p of this.waiting.values()) p.reject(new Error(event.message || 'Layout worker failed'));
      this.waiting.clear();
    };
  }

  partition(pngBuffer) {
    return new Promise((resolve, reject) => {
      const id = ++this.nextId;
      this.waiting.set(id, {resolve, reject});
      this.worker.postMessage({id, png: pngBuffer}, [pngBuffer]);
    });
  }
}
