"""Runs Bina 0.4 Koochik (ONNX-INT4) with ONNX Runtime on an NVIDIA GPU (CUDA) or the CPU.

Follows the reference runtime of Reza2kn/Bina-0.4-Koochik-ONNX-INT4
(bina_portable.py, onnx_gpu_worker.py, portable_route.py; OpenMDW-1.1):
greedy decoding with repetition penalty 1.1, at most 2,048 tokens per block,
one higher-resolution retry when the output is capped or repeats, TF32 off on
CUDA and the decoder's keys/values kept on the GPU between steps.
"""
import collections
import json
import os
import pathlib
import time

import numpy as np

from . import preprocess as P

PROMPTS = {
    'text': [2, 0, 50009],               # </s><s><predict_text_in_pic>
    'markdown': [2, 0, 50001, 50009],    # </s><s><output_markdown><predict_text_in_pic>
}
MAX_NEW_TOKENS = 2048
COMPONENTS = ('vision', 'neck', 'prefill', 'decode')


class Cancelled(Exception):
    pass


class GpuFailure(RuntimeError):
    pass


def repetition_signal(ids, plain, eos):
    ids = [x for x in ids if x != eos]
    counts = collections.Counter(tuple(ids[i:i + 4]) for i in range(len(ids) - 3))
    dominant = max(counts.values(), default=0)
    token_loop = len(ids) >= 64 and dominant >= 12 and 4 * dominant / max(1, len(ids)) >= .35
    longest = run = 0
    last = None
    for c in plain:
        run = run + 1 if c == last else 1
        last = c
        if not c.isspace():
            longest = max(longest, run)
    return token_loop or longest >= 32


def add_cuda_dll_dirs(cuda_dir):
    """Make the NVIDIA DLLs in cuda_dir loadable (Windows) and preload them for ORT."""
    import onnxruntime as ort
    cuda_dir = pathlib.Path(cuda_dir)
    dirs = sorted({str(p.parent) for p in cuda_dir.rglob('*.dll')})
    if os.name == 'nt':
        for d in dirs:
            os.add_dll_directory(d)
        os.environ['PATH'] = os.pathsep.join(dirs + [os.environ.get('PATH', '')])
    # Loads cudart/cublas/cufft/cuDNN from cuda_dir/nvidia/... (same layout as NVIDIA's wheels).
    ort.preload_dlls(directory=str(cuda_dir))


class Engine:
    def __init__(self, model_dir, device='auto', cuda_dir=None, log=print):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.ort = ort
        self.log = log
        root = pathlib.Path(model_dir)
        assets = root / 'runtime-assets'
        self.onnx_dir = root / 'onnx'
        self.config = json.loads((assets / 'preprocessor_config.json').read_text('utf-8'))
        generation = json.loads((assets / 'generation_config.json').read_text('utf-8'))
        self.eos = generation['eos_token_id']
        self.forced_eos = generation.get('forced_eos_token_id', self.eos)
        self.penalty = generation.get('repetition_penalty', 1.1)
        self.tokenizer = Tokenizer.from_file(str(assets / 'tokenizer.json'))
        self.field = P.load_field(assets / 'position-field.npy')
        self.sessions = {}
        self.device = 'cpu'
        self.fallback_reason = None
        want_gpu = device in ('auto', 'cuda')
        if want_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
            try:
                if cuda_dir:
                    add_cuda_dll_dirs(cuda_dir)
                self._load('CUDAExecutionProvider')
                self.device = 'cuda'
                self._warm_up()
            except Exception as error:  # driver/DLL problems: use the CPU instead
                self.fallback_reason = str(error)
                log('CUDA unavailable, using CPU: ' + str(error))
                if device == 'cuda':
                    raise
                self.sessions = {}
                self.device = 'cpu'
        elif device == 'cuda':
            raise GpuFailure('CUDAExecutionProvider is not available')
        if self.device == 'cpu':
            self._load('CPUExecutionProvider')

    # -- sessions ---------------------------------------------------------
    def _load(self, provider):
        ort = self.ort
        self.sessions = {}
        for name in COMPONENTS:
            options = ort.SessionOptions()
            options.log_severity_level = 3
            if provider == 'CUDAExecutionProvider':
                providers = [(provider, {'device_id': 0, 'use_tf32': '0'}), 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            session = ort.InferenceSession(str(self.onnx_dir / f'{name}.int4.onnx'), sess_options=options, providers=providers)
            if provider not in session.get_providers():
                raise GpuFailure('ONNX Runtime could not start ' + provider)
            self.sessions[name] = session

    def _warm_up(self):
        """A short real run, so that a broken GPU setup is found before the user's files."""
        rgb = np.full((48, 256, 3), 255, np.uint8)
        rgb[18:30, 40:216] = 0
        self._attempt(rgb, PROMPTS['text'], max_new_tokens=4)

    # -- model steps ------------------------------------------------------
    def _encode(self, rgb):
        pre = P.process_pixels(rgb, self.config)
        pos = P.positions(self.field, pre['content_size'], pre['image_size'])
        count = pre['rows'] * pre['cols']
        features, summary = self.sessions['vision'].run(None, {
            'patches': pre['patches'], 'positions': pos.reshape(1, count, 1152)})
        grid = np.ascontiguousarray(features.reshape(1, count, 1152).transpose(0, 2, 1).reshape(1, 1152, pre['rows'], pre['cols']))
        return self.sessions['neck'].run(None, {'features': grid, 'summary': summary})[0]

    def _prefill(self, ids, vision):
        token_ids = np.array([ids], np.int64)
        if self.device == 'cuda':
            session = self.sessions['prefill']
            b = session.io_binding()
            b.bind_cpu_input('token_ids', token_ids)
            b.bind_cpu_input('vision', np.ascontiguousarray(vision))
            b.bind_output('logits', 'cpu')
            b.bind_output('keys', 'cuda', 0)
            b.bind_output('values', 'cuda', 0)
            session.run_with_iobinding(b)
            out = b.get_outputs()
            return out[0].numpy(), (out[1], out[2])
        logits, keys, values = self.sessions['prefill'].run(None, {'token_ids': token_ids, 'vision': vision})
        return logits, (keys, values)

    def _decode(self, token, cache):
        token_ids = np.array([[token]], np.int64)
        if self.device == 'cuda':
            session = self.sessions['decode']
            b = session.io_binding()
            b.bind_cpu_input('token_ids', token_ids)
            b.bind_ortvalue_input('past_keys', cache[0])
            b.bind_ortvalue_input('past_values', cache[1])
            b.bind_output('logits', 'cpu')
            b.bind_output('keys', 'cuda', 0)
            b.bind_output('values', 'cuda', 0)
            session.run_with_iobinding(b)
            out = b.get_outputs()
            return out[0].numpy(), (out[1], out[2])
        logits, keys, values = self.sessions['decode'].run(None, {'token_ids': token_ids, 'past_keys': cache[0], 'past_values': cache[1]})
        return logits, (keys, values)

    def _attempt(self, rgb, prompt, max_new_tokens=MAX_NEW_TOKENS, on_token=None, cancel=None):
        vision = self._encode(rgb)
        ids = list(prompt)
        logits, cache = self._prefill(ids, vision)
        seen = set(ids)
        generated = []
        for i in range(max_new_tokens):
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            scores = np.array(logits, np.float32).reshape(-1)
            if not np.isfinite(scores).all():
                raise GpuFailure('Non-finite logits') if self.device == 'cuda' else RuntimeError('Non-finite logits')
            idx = np.fromiter(seen, np.int64)
            s = scores[idx]
            scores[idx] = np.where(s < 0, s * self.penalty, s / self.penalty)
            token = int(scores.argmax()) if i < max_new_tokens - 1 else self.forced_eos
            generated.append(token)
            seen.add(token)
            if token == self.eos:
                break
            if on_token is not None and (len(generated) % 16 == 0):
                on_token(len(generated), generated)
            logits, cache = self._decode(token, cache)
        cache = None
        plain = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        body = self.tokenizer.decode(generated, skip_special_tokens=False)
        if body.endswith('</s>'):
            body = body[:-4]
        return {
            'ids': generated, 'text': plain, 'markup': body, 'tokens': len(generated),
            'capped': len(generated) >= max_new_tokens,
            'eos': bool(generated and generated[-1] == self.eos),
            'flagged': repetition_signal(generated, plain, self.eos),
        }

    def decode_text(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def transcribe(self, rgb, mode='text', on_progress=None, cancel=None):
        """rgb: uint8 (h, w, 3). Returns the selected attempt (dict with 'text')."""
        prompt = PROMPTS[mode]
        try:
            return self._transcribe(rgb, prompt, on_progress, cancel)
        except GpuFailure as error:
            # Same policy as the reference runtime: redo the whole block on the CPU.
            self.log('GPU failed, switching to CPU: ' + str(error))
            self.fallback_reason = str(error)
            self.sessions = {}
            self.device = 'cpu'
            self._load('CPUExecutionProvider')
            return self._transcribe(rgb, prompt, on_progress, cancel)

    def _transcribe(self, rgb, prompt, on_progress, cancel):
        def stage(name, tokens=0, ids=None):
            if on_progress:
                on_progress(name, tokens, ids)
        original = rgb
        primary = P.primary_image(original)
        stage('vision')
        first = self._attempt(primary, prompt, on_token=lambda n, ids: stage('decode', n, ids), cancel=cancel)
        if first['capped'] or first['flagged']:
            retry_rgb = P.retry_image(original, primary)
            if retry_rgb is not None:
                stage('retry')
                second = self._attempt(retry_rgb, prompt, on_token=lambda n, ids: stage('decode', n, ids), cancel=cancel)
                if second['text'] and not second['flagged'] and second['eos'] and not second['capped']:
                    return second
        return first

    def close(self):
        self.sessions = {}
