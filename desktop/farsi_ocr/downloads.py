"""Downloads the Bina model and (optionally) NVIDIA's CUDA libraries, with resume and hash checks."""
import hashlib
import os
import pathlib
import sys
import urllib.request
import zipfile

APP_NAME = 'FarsiOCR'

MODEL_REPO = 'Reza2kn/Bina-0.4-Koochik-ONNX-INT4'
MODEL_REVISION = '89e20a65255511bf28442955aad429688083874c'
MODEL_URL = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/'
# path, size, hash (sha256 for large files, git blob sha1 for small ones)
MODEL_FILES = [
    ('LICENSE', 3051, '3618e26f595c524d2aee87535e8ed4257cfd1adb'),
    ('runtime-assets/generation_config.json', 286, '650b97aba9eb5d93646d19fc790124d69fa9669f'),
    ('runtime-assets/preprocessor_config.json', 593, '5c3665b81707c3b946ce9dc60cd1669504da03fb'),
    ('runtime-assets/tokenizer.json', 7627412, 'cc5efa9e6dccd31cc047158e05a7523dd365c74e'),
    ('runtime-assets/position-field.npy', 75497600, '082f0ee1ffec0798c5c65718006f2002191f88f8ec334c44e78a39e175af39b6'),
    ('onnx/neck.int4.onnx', 3941, '2f5512c7d9dd56c58c0e0cd9849905f27b78c926a5923390992d1c349472cc8b'),
    ('onnx/neck.int4.onnx.data', 23003136, '2479030719fefb30865298b21652c822c2f9d8d9437fd09902f92575eadff247'),
    ('onnx/vision.int4.onnx', 433349, '294b1c3e53a960a2ffc3dee0a65e86c648a9dab04d2d4f1ff82ead97c8442545'),
    ('onnx/vision.int4.onnx.data', 259413696, '1398dcad4bc606faf546f86c948750ad327f7f7422121d2d108fc8ae7da0a03a'),
    ('onnx/prefill.int4.onnx', 144191, '16bf81ea849213fab9ff41a7dcc86c88412c2b0c2c03e02946c191c50ea6a1f2'),
    ('onnx/decode.int4.onnx', 147915, 'fd7774f150816e322f3acc450048727d48807dc2c703e0647dad7a8cd38d40fa'),
    ('onnx/decoder.int4.onnx.data', 369606656, 'b6256ce14eca54e514b9a3e4abcc892e898f8a7ac61fa877e69148419623ea49'),
]
MODEL_BYTES = sum(size for _, size, _ in MODEL_FILES)

# NVIDIA CUDA 13 + cuDNN 9 runtime DLLs for onnxruntime-gpu 1.31 (from NVIDIA's PyPI wheels).
CUDA_WHEELS = [
    ('nvidia-cuda-runtime', 'https://files.pythonhosted.org/packages/35/99/e07dd9565d2e0317c1a62ff39ae9fc0542c048f74bcfc777c8be97710347/nvidia_cuda_runtime-13.5.37-py3-none-win_amd64.whl', 'f225d3fa5ea6bfbb58fc4979ef72375cbe661d9107f0d8a4ff22cc151525e4a9', 2780254),
    ('nvidia-cuda-nvrtc', 'https://files.pythonhosted.org/packages/9f/6c/241c5155199a6de55bad8862213ed64b487dc9526aae817e4af31afea1a4/nvidia_cuda_nvrtc-13.5.37-py3-none-win_amd64.whl', 'e8e6c10b251385707ae3abd7b4ee118de6736fd7678d0f0f4eab4d759f679c21', 49248991),
    ('nvidia-nvjitlink', 'https://files.pythonhosted.org/packages/a7/53/c43398492e8374880d5e01c23c4dac33ba0c3bd03515d6ecc574254c0967/nvidia_nvjitlink-13.5.37-py3-none-win_amd64.whl', '4465d4e8f2212118220b580c82523c5654f93a62906bd51d29d79bdfc12376c1', 41467665),
    ('nvidia-cufft', 'https://files.pythonhosted.org/packages/94/79/8f2a7125f560b86571c0be3dbf10e33041187bc4a8cf66107541fe8e5cce/nvidia_cufft-12.5.0.14-py3-none-win_amd64.whl', '78e7585159ecf37d73af1ce43a5dea7fd3360e1af71cde08f070c07966a336bc', 164476208),
    ('nvidia-cudnn-cu13', 'https://files.pythonhosted.org/packages/48/9c/26e6bdd31c97c389aa565b49c916be0dec17dd61dad6cd33fee286d1b946/nvidia_cudnn_cu13-9.28.1.4-py3-none-win_amd64.whl', '73d874c6bd8a1c75ca040228f77c4e163efe9bab3b65bd85914e2b7334ac68ee', 407065099),
    ('nvidia-cublas', 'https://files.pythonhosted.org/packages/12/39/e1b1b4adb8b171894d9a89035df0596d448cf73dc403802d24dbe30cc0be/nvidia_cublas-13.10.0.20-py3-none-win_amd64.whl', 'c310b749eea855b88fafb906de08d6819b87216c6711640771764fea5dcd8546', 427284863),
]
CUDA_BYTES = sum(w[3] for w in CUDA_WHEELS)
CUDA_PACK_VERSION = 'cuda13.5-cudnn9.28'


class Cancelled(Exception):
    pass


def data_dir():
    if os.name == 'nt':
        base = pathlib.Path(os.environ.get('LOCALAPPDATA') or pathlib.Path.home() / 'AppData' / 'Local')
    elif sys.platform == 'darwin':
        base = pathlib.Path.home() / 'Library' / 'Application Support'
    else:
        base = pathlib.Path(os.environ.get('XDG_DATA_HOME') or pathlib.Path.home() / '.local' / 'share')
    return base / APP_NAME


def model_dir():
    return data_dir() / 'models' / f'bina-0.4-koochik-onnx-int4-{MODEL_REVISION[:8]}'


def cuda_dir():
    return data_dir() / 'cuda' / CUDA_PACK_VERSION


def _git_blob_sha1(path):
    h = hashlib.sha1()
    h.update(b'blob %d\0' % path.stat().st_size)
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _matches(path, size, digest):
    if not path.is_file() or path.stat().st_size != size:
        return False
    return (_sha256(path) if len(digest) == 64 else _git_blob_sha1(path)) == digest


def model_ready(root=None):
    root = pathlib.Path(root or model_dir())
    if (root / '.verified').is_file():
        return all((root / p).is_file() and (root / p).stat().st_size == s for p, s, _ in MODEL_FILES)
    return False


def cuda_ready(root=None):
    return (pathlib.Path(root or cuda_dir()) / '.complete').is_file()


def _fetch(url, dest, size, done_before, total, progress, cancel):
    """Downloads url to dest with resume. progress(done_bytes, total_bytes)."""
    part = dest.with_name(dest.name + '.part')
    have = part.stat().st_size if part.exists() else 0
    if have > size:
        part.unlink()
        have = 0
    if have < size:
        request = urllib.request.Request(url, headers={'User-Agent': 'FarsiOCR-desktop'})
        if have:
            request.add_header('Range', f'bytes={have}-')
        with urllib.request.urlopen(request, timeout=60) as response:
            if have and response.status != 206:
                have = 0  # server ignored the range: start over
            with open(part, 'ab' if have else 'wb') as out:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled()
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    have += len(chunk)
                    if progress:
                        progress(done_before + have, total)
    return part


def download_model(progress=None, cancel=None, root=None):
    root = pathlib.Path(root or model_dir())
    root.mkdir(parents=True, exist_ok=True)
    done = 0
    for path, size, digest in MODEL_FILES:
        dest = root / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not _matches(dest, size, digest):
            part = _fetch(MODEL_URL + path, dest, size, done, MODEL_BYTES, progress, cancel)
            if not _matches(part, size, digest):
                part.unlink(missing_ok=True)
                raise IOError(f'Downloaded file is damaged: {path}. Please try again.')
            os.replace(part, dest)
        done += size
        if progress:
            progress(done, MODEL_BYTES)
    (root / '.verified').write_text(MODEL_REVISION)
    return root


def download_cuda(progress=None, cancel=None, root=None):
    """Fetches NVIDIA's wheels and keeps only their DLLs, in the nvidia/... layout ORT expects."""
    root = pathlib.Path(root or cuda_dir())
    downloads = root.parent / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    done = 0
    for name, url, digest, size in CUDA_WHEELS:
        wheel = downloads / url.rsplit('/', 1)[1]
        if not (wheel.is_file() and wheel.stat().st_size == size):
            part = _fetch(url, wheel, size, done, CUDA_BYTES, progress, cancel)
            if _sha256(part) != digest:
                part.unlink(missing_ok=True)
                raise IOError(f'Downloaded file is damaged: {name}. Please try again.')
            os.replace(part, wheel)
        with zipfile.ZipFile(wheel) as z:
            for info in z.infolist():
                if info.filename.lower().endswith('.dll') and info.filename.startswith('nvidia/'):
                    z.extract(info, root)
        wheel.unlink()
        done += size
        if progress:
            progress(done, CUDA_BYTES)
    (root / '.complete').write_text(CUDA_PACK_VERSION)
    return root


def cuda_dll_dirs(root=None):
    root = pathlib.Path(root or cuda_dir())
    return sorted({str(p.parent) for p in root.rglob('*.dll')})


def has_nvidia_gpu():
    """True when an NVIDIA driver is installed (nvcuda.dll on Windows)."""
    if os.name != 'nt':
        return False
    import ctypes
    try:
        ctypes.WinDLL('nvcuda.dll')
        return True
    except OSError:
        return False


def resource_path(*parts):
    """A bundled file, both from source and inside the PyInstaller build."""
    base = pathlib.Path(getattr(sys, '_MEIPASS', pathlib.Path(__file__).resolve().parent.parent))
    return base.joinpath(*parts)
