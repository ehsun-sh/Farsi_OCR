"""Command-line mode (also used for testing the Windows build).

  FarsiOCR.exe --check report.txt
  FarsiOCR.exe --ocr scan.pdf photo.jpg --out result.docx [--device cpu|cuda|auto] [--mode text|markdown]
               [--pages 1-3] [--download] [--log run.log]
"""
import argparse
import pathlib
import platform
import sys
import time

from . import __version__, downloads, exporters


def _writer(path):
    stream = open(path, 'a', encoding='utf-8') if path else (sys.stderr or open('farsi-ocr.log', 'a', encoding='utf-8'))

    def log(*parts):
        stream.write(' '.join(str(p) for p in parts) + '\n')
        stream.flush()
    return log


def check(report):
    lines = [f'FarsiOCR {__version__}', f'python {sys.version}', f'platform {platform.platform()}']
    try:
        import onnxruntime as ort
        lines.append(f'onnxruntime {ort.__version__} providers={ort.get_available_providers()}')
        import numpy, tokenizers, PIL, pypdfium2, docx  # noqa: F401
        from PySide6 import __version__ as qt
        lines.append(f'numpy {numpy.__version__} tokenizers {tokenizers.__version__} pillow {PIL.__version__} pyside6 {qt}')
        from . import engine, layout, preprocess  # noqa: F401
        import numpy as np
        rgb = np.full((300, 900, 3), 255, np.uint8)
        rgb[100:120, 50:850] = 0
        lines.append(f'layout blocks {layout.blocks(rgb)}')
        pre = preprocess.process_pixels(rgb, {'image_mean': [.5] * 3, 'image_std': [.5] * 3})
        lines.append(f"preprocess grid {pre['rows']}x{pre['cols']}")
        lines.append(f'nvidia driver {downloads.has_nvidia_gpu()} model_ready {downloads.model_ready()} cuda_ready {downloads.cuda_ready()}')
        lines.append('OK')
        code = 0
    except Exception as error:
        import traceback
        lines.append('FAILED ' + repr(error))
        lines.append(traceback.format_exc())
        code = 1
    pathlib.Path(report).write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return code


def main(argv):
    p = argparse.ArgumentParser(prog='FarsiOCR')
    p.add_argument('--check')
    p.add_argument('--ocr', nargs='+')
    p.add_argument('--out')
    p.add_argument('--device', default='cpu', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--mode', default='text', choices=['text', 'markdown'])
    p.add_argument('--pages', default='')
    p.add_argument('--pdf-width', type=int, default=1240)
    p.add_argument('--download', action='store_true', help='download the model (and CUDA libraries if --device cuda)')
    p.add_argument('--log')
    a = p.parse_args(argv)
    if a.check:
        return check(a.check)
    log = _writer(a.log)
    if a.download or not downloads.model_ready():
        log('checking/downloading the model')
        last = [0]

        def progress(done, total):
            if done - last[0] > 50e6 or done == total:
                last[0] = done
                log(f'download/verify {done / 1e6:.0f}/{total / 1e6:.0f} MB')
        downloads.download_model(progress)
        if a.device == 'cuda' and not downloads.cuda_ready():
            downloads.download_cuda(progress)
    from . import layout
    from .documents import Document
    from .engine import Engine
    tick = time.monotonic()
    cuda = str(downloads.cuda_dir()) if downloads.cuda_ready() else None
    engine = Engine(downloads.model_dir(), device=a.device, cuda_dir=cuda, log=log)
    log(f'engine ready on {engine.device} in {time.monotonic() - tick:.1f}s' + (f' (fallback: {engine.fallback_reason})' if engine.fallback_reason else ''))
    docs = []
    for path in a.ocr:
        doc = Document(path)
        pages = []
        for number in doc.page_numbers(a.pages):
            t = time.monotonic()
            rgb = doc.render(number, a.pdf_width)
            texts = []
            blocks = layout.blocks(rgb)
            for block in blocks:
                x0, y0, x1, y1 = block['bbox']
                bt = time.monotonic()
                result = engine.transcribe(rgb[y0:y1, x0:x1], a.mode)
                log(f'  block {block["bbox"]} patches={block["patches"]} tokens={result["tokens"]} {time.monotonic() - bt:.1f}s')
                if result['text']:
                    texts.append(result['text'])
            pages.append({'number': number, 'text': '\n\n'.join(texts)})
            log(f'{doc.name} page {number}: {len(blocks)} blocks, {time.monotonic() - t:.1f}s')
        docs.append({'name': doc.name, 'is_pdf': doc.is_pdf, 'mode': a.mode, 'pages': pages})
    out = pathlib.Path(a.out or (exporters.base_name(docs) + '.txt'))
    if out.suffix.lower() == '.docx':
        exporters.to_docx(docs, out)
    elif out.suffix.lower() == '.md':
        out.write_text(exporters.to_markdown(docs), encoding='utf-8')
    else:
        out.write_text(exporters.to_text(docs), encoding='utf-8')
    log(f'done in {time.monotonic() - tick:.1f}s on {engine.device} -> {out}')
    return 0
