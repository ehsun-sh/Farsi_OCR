"""Desktop window (PySide6, right-to-left Persian UI)."""
import math
import pathlib
import sys
import threading
import time
import traceback

from PySide6.QtCore import QObject, QSettings, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QDesktopServices, QFont, QFontDatabase, QGuiApplication, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QProgressBar,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from . import __version__, downloads, exporters
from .documents import Document, is_supported

FA = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')
BINA_URL = 'https://huggingface.co/Reza2kn/Bina-0.4-Koochik'
ONNX_URL = 'https://huggingface.co/Reza2kn/Bina-0.4-Koochik-ONNX-INT4'
SITE_URL = 'https://farsi-ocr.ehsun-ca.workers.dev/'
REPO_URL = 'https://github.com/ehsun-sh/Farsi_OCR'
# first-guess speed (seconds per 16x16 image patch) until this computer has been measured
DEFAULT_RATE = {'cuda': 1.5 / 1300, 'cpu': 6 / 1300}


def fa(n):
    return str(n).translate(FA)


def fmt_mb(n):
    return fa(f'{n / 1e6:,.0f}'.replace(',', '٬')) + ' مگابایت'


def fmt_duration(seconds):
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f'{fa(max(seconds, 1))} ثانیه'
    minutes = round(seconds / 60)
    if minutes < 60:
        return f'{fa(minutes)} دقیقه'
    hours, minutes = divmod(minutes, 60)
    return f'{fa(hours)} ساعت' + (f' و {fa(minutes)} دقیقه' if minutes else '')


# ---------------------------------------------------------------------------
# Background work

class Worker(QObject):
    """Runs downloads, model loading and OCR off the UI thread."""
    status = Signal(str)
    download_progress = Signal(int, int)
    model_loaded = Signal(str, str)        # device, fallback reason
    failed = Signal(str, str)              # where, message
    finished_download = Signal(str)        # 'model' | 'cuda'
    run_progress = Signal(dict)
    page_done = Signal(int, int, str)      # doc index, page number, text
    run_finished = Signal(bool)            # completed (False when stopped)

    def __init__(self):
        super().__init__()
        self.engine = None
        self.engine_device = None
        self.cancel = threading.Event()

    def _log(self, message):
        self.status.emit(message)

    @Slot(str)
    def download(self, what):
        self.cancel.clear()
        try:
            if what == 'model':
                downloads.download_model(lambda d, t: self.download_progress.emit(d // 1024, t // 1024), self.cancel)
            else:
                downloads.download_cuda(lambda d, t: self.download_progress.emit(d // 1024, t // 1024), self.cancel)
            self.finished_download.emit(what)
        except downloads.Cancelled:
            self.failed.emit(what, 'دانلود متوقف شد. دفعهٔ بعد از همان‌جا ادامه پیدا می‌کند.')
        except Exception as error:
            self.failed.emit(what, f'دانلود ناموفق بود: {error}')

    @Slot(str)
    def load_engine(self, device):
        try:
            from .engine import Engine
            if self.engine is not None and self.engine_device == device:
                self.model_loaded.emit(self.engine.device, self.engine.fallback_reason or '')
                return
            self.engine = None
            cuda = str(downloads.cuda_dir()) if downloads.cuda_ready() else None
            self.status.emit('در حال بالا آوردن مدل…')
            self.engine = Engine(downloads.model_dir(), device=device, cuda_dir=cuda, log=self._log)
            self.engine_device = device
            self.model_loaded.emit(self.engine.device, self.engine.fallback_reason or '')
        except Exception as error:
            traceback.print_exc()
            self.failed.emit('engine', f'بالا آوردن مدل ناموفق بود: {error}')

    @Slot(list, dict)
    def run(self, paths, options):
        from . import layout
        from .engine import Cancelled
        self.cancel.clear()
        completed = False
        try:
            docs = [Document(p) for p in paths]
            plan = [(i, d, d.page_numbers(options['pages'])) for i, d in enumerate(docs)]
            total_pages = sum(len(n) for _, _, n in plan)
            done_pages = 0
            for i, doc, numbers in plan:
                for number in numbers:
                    if self.cancel.is_set():
                        raise Cancelled()
                    label = f'{doc.name}' + (f' — صفحهٔ {fa(number)}' if doc.is_pdf or doc.page_count > 1 else '')
                    self.run_progress.emit({'stage': 'render', 'label': label})
                    rgb = doc.render(number, options['pdf_width'])
                    blocks = layout.blocks(rgb)
                    self.run_progress.emit({'stage': 'page', 'patches': [b['patches'] for b in blocks],
                                            'total_pages': total_pages, 'label': label})
                    texts = []
                    for k, block in enumerate(blocks):
                        x0, y0, x1, y1 = block['bbox']
                        started = time.monotonic()
                        self.run_progress.emit({'stage': 'block', 'index': k, 'count': len(blocks), 'label': label,
                                                'patches': block['patches']})

                        def on_progress(stage, tokens, ids, k=k):
                            self.run_progress.emit({'stage': stage, 'tokens': tokens, 'index': k, 'count': len(blocks), 'label': label})
                        result = self.engine.transcribe(rgb[y0:y1, x0:x1], options['mode'], on_progress, self.cancel)
                        if result['text']:
                            texts.append(result['text'])
                        self.run_progress.emit({'stage': 'block-done', 'patches': block['patches'],
                                                'seconds': time.monotonic() - started, 'device': self.engine.device})
                        self.page_done.emit(i, number, '\n\n'.join(texts))
                    if not blocks:
                        self.page_done.emit(i, number, '')
                    done_pages += 1
            completed = True
        except Cancelled:
            pass
        except Exception as error:
            traceback.print_exc()
            self.failed.emit('run', f'خطا در تشخیص متن: {error}')
        self.run_finished.emit(completed)


class Progress:
    def __init__(self, total_pages, rate):
        self.total_pages = total_pages
        self.pages = 0
        self.known = 0
        self.done = 0
        self.seconds = 0.0
        self.current = 0.0
        self.current_patches = 0
        self.default_rate = rate

    def add_page(self, patches):
        self.pages += 1
        self.known += sum(patches)

    @property
    def total(self):
        per_page = self.known / self.pages if self.pages else 0
        return max(1, self.known + (self.total_pages - self.pages) * per_page)

    @property
    def rate(self):
        return self.seconds / self.done if self.done else self.default_rate

    @property
    def fraction(self):
        return min(1.0, (self.done + self.current * self.current_patches) / self.total)

    @property
    def remaining(self):
        return (self.total - self.done - self.current * self.current_patches) * self.rate


def block_fraction(stage, tokens):
    if stage in ('vision', 'block'):
        return 0.05
    if stage == 'retry':
        return 0.5
    if stage == 'decode':
        return 0.3 + 0.65 * (1 - math.exp(-tokens / 120))
    return 0.0


# ---------------------------------------------------------------------------
# Window

class DropList(QListWidget):
    files_dropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setMinimumHeight(70)
        self.setMaximumHeight(160)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        self.files_dropped.emit(paths)
        e.acceptProposedAction()


class MainWindow(QMainWindow):
    request_download = Signal(str)
    request_engine = Signal(str)
    request_run = Signal(list, dict)

    def __init__(self):
        super().__init__()
        self.settings = QSettings('FarsiOCR', 'FarsiOCR')
        self.setWindowTitle(f'OCR فارسی — v{__version__}')
        self.resize(980, 820)
        self.paths = []
        self.results = {}      # (doc index, page) -> QPlainTextEdit
        self.result_docs = []
        self.busy = None       # 'download-model' | 'download-cuda' | 'engine' | 'run'
        self.device = None
        self.progress = None

        self.thread = QThread(self)
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.start()
        self.request_download.connect(self.worker.download)
        self.request_engine.connect(self.worker.load_engine)
        self.request_run.connect(self.worker.run)
        self.worker.status.connect(self.set_status)
        self.worker.download_progress.connect(self.on_download_progress)
        self.worker.finished_download.connect(self.on_download_done)
        self.worker.model_loaded.connect(self.on_model_loaded)
        self.worker.failed.connect(self.on_failed)
        self.worker.run_progress.connect(self.on_run_progress)
        self.worker.page_done.connect(self.on_page_done)
        self.worker.run_finished.connect(self.on_run_finished)

        self._build()
        self.refresh_model_box()
        if downloads.model_ready():
            self.load_engine()

    # -- layout -----------------------------------------------------------
    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget()
        scroll.setWidget(body)
        self.setCentralWidget(scroll)
        v = QVBoxLayout(body)
        v.setContentsMargins(18, 14, 18, 14)
        v.setSpacing(12)

        head = QHBoxLayout()
        logo_path = downloads.resource_path('assets', 'logo.png')
        if logo_path.is_file():
            logo = QLabel()
            logo.setPixmap(QPixmap(str(logo_path)).scaledToHeight(40, Qt.SmoothTransformation))
            head.addWidget(logo)
        title = QLabel('Farsi OCR')
        title.setObjectName('title')
        head.addWidget(title)
        head.addStretch(1)
        v.addLayout(head)
        lead = QLabel(
            'عکس یا PDF فارسی را بدهید و متن را به‌صورت <b>TXT</b>، <b>Word</b> یا <b>Markdown</b> بگیرید. '
            f'تشخیص متن با مدل <a href="{BINA_URL}">بینا (Bina 0.4 Koochik)</a> انجام می‌شود؛ '
            'همه‌چیز روی همین کامپیوتر اجرا می‌شود و هیچ فایلی جایی فرستاده نمی‌شود.')
        lead.setWordWrap(True)
        lead.setOpenExternalLinks(True)
        v.addWidget(lead)

        # Step 1: model
        box = QGroupBox('۱ · مدل')
        g = QVBoxLayout(box)
        self.model_label = QLabel()
        self.model_label.setWordWrap(True)
        self.model_label.setTextFormat(Qt.RichText)
        g.addWidget(self.model_label)
        row = QHBoxLayout()
        self.btn_model = QPushButton('دانلود مدل')
        self.btn_model.setObjectName('primary')
        self.btn_model.clicked.connect(lambda: self.start_download('model'))
        self.btn_cuda = QPushButton('دانلود CUDA برای کارت NVIDIA (آزمایشی)')
        self.btn_cuda.clicked.connect(lambda: self.start_download('cuda'))
        self.btn_cancel_dl = QPushButton('توقف دانلود')
        self.btn_cancel_dl.clicked.connect(self.worker.cancel.set)
        row.addWidget(QLabel('پردازش با:'))
        self.device_box = QComboBox()
        self.device_box.addItem('CPU (پیشنهادی)', 'cpu')
        self.device_box.addItem('کارت گرافیک NVIDIA (آزمایشی)', 'cuda')
        # 0.5.0 defaulted to 'auto' (GPU first), which turned out much slower on some laptops
        saved = self.settings.value('device-v2', 'cpu')
        self.device_box.setCurrentIndex(max(0, self.device_box.findData(saved)))
        self.device_box.currentIndexChanged.connect(self.on_device_changed)
        row.addWidget(self.device_box)
        row.addStretch(1)
        row.addWidget(self.btn_cuda)
        row.addWidget(self.btn_model)
        row.addWidget(self.btn_cancel_dl)
        g.addLayout(row)
        self.dl_bar = QProgressBar()
        self.dl_bar.setVisible(False)
        g.addWidget(self.dl_bar)
        v.addWidget(box)

        # Step 2: files
        box = QGroupBox('۲ · فایل‌ها')
        g = QVBoxLayout(box)
        hint = QLabel('فایل‌ها را اینجا بکشید و رها کنید یا «افزودن فایل» را بزنید. PNG، JPG، WebP، TIFF و PDF.')
        hint.setWordWrap(True)
        hint.setObjectName('muted')
        g.addWidget(hint)
        self.file_list = DropList()
        self.file_list.files_dropped.connect(self.add_files)
        g.addWidget(self.file_list)
        row = QHBoxLayout()
        b = QPushButton('افزودن فایل…')
        b.clicked.connect(self.pick_files)
        row.addWidget(b)
        b = QPushButton('حذف انتخاب‌شده‌ها')
        b.clicked.connect(self.remove_selected)
        row.addWidget(b)
        row.addStretch(1)
        g.addLayout(row)

        grid = QGridLayout()
        grid.addWidget(QLabel('نوع خروجی مدل'), 0, 0)
        self.mode_box = QComboBox()
        self.mode_box.addItem('متن ساده (پیشنهادی)', 'text')
        self.mode_box.addItem('Markdown با عنوان و جدول (آزمایشی)', 'markdown')
        grid.addWidget(self.mode_box, 1, 0)
        grid.addWidget(QLabel('صفحه‌های PDF'), 0, 1)
        self.pages_edit = QLineEdit()
        self.pages_edit.setPlaceholderText('همه؛ مثلا 1-3, 5')
        grid.addWidget(self.pages_edit, 1, 1)
        grid.addWidget(QLabel('کیفیت رندر PDF'), 0, 2)
        self.quality_box = QComboBox()
        self.quality_box.addItem('سریع', 1000)
        self.quality_box.addItem('معمولی (پیشنهادی)', 1240)
        self.quality_box.addItem('بالا (برای حروف ریز، کندتر)', 1650)
        self.quality_box.setCurrentIndex(1)
        grid.addWidget(self.quality_box, 1, 2)
        g.addLayout(grid)

        row = QHBoxLayout()
        self.btn_start = QPushButton('شروع تشخیص متن')
        self.btn_start.setObjectName('primary')
        self.btn_start.clicked.connect(self.start_run)
        self.btn_stop = QPushButton('توقف')
        self.btn_stop.clicked.connect(self.worker.cancel.set)
        row.addWidget(self.btn_start)
        row.addWidget(self.btn_stop)
        row.addStretch(1)
        g.addLayout(row)
        self.run_bar = QProgressBar()
        self.run_bar.setRange(0, 1000)
        self.run_bar.setTextVisible(False)
        self.run_bar.setVisible(False)
        g.addWidget(self.run_bar)
        self.eta_label = QLabel()
        self.eta_label.setObjectName('eta')
        self.eta_label.setVisible(False)
        g.addWidget(self.eta_label)
        self.status_label = QLabel()
        self.status_label.setObjectName('muted')
        self.status_label.setWordWrap(True)
        g.addWidget(self.status_label)
        v.addWidget(box)

        # Step 3: results
        self.results_box = QGroupBox('۳ · نتیجه')
        g = QVBoxLayout(self.results_box)
        note = QLabel('متن را می‌توانید همین‌جا ویرایش کنید؛ فایل‌های خروجی از متن ویرایش‌شده ساخته می‌شوند.')
        note.setObjectName('muted')
        note.setWordWrap(True)
        g.addWidget(note)
        row = QHBoxLayout()
        for label, kind in (('ذخیره Word (.docx)', 'docx'), ('ذخیره متن (.txt)', 'txt'), ('ذخیره Markdown (.md)', 'md')):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self.export(k))
            row.addWidget(b)
        b = QPushButton('کپی همهٔ متن')
        b.clicked.connect(self.copy_all)
        row.addWidget(b)
        row.addStretch(1)
        b = QPushButton('پاک کردن نتایج')
        b.clicked.connect(self.clear_results)
        row.addWidget(b)
        g.addLayout(row)
        self.results_area = QVBoxLayout()
        g.addLayout(self.results_area)
        self.results_box.setVisible(False)
        v.addWidget(self.results_box)

        v.addStretch(1)
        foot = QLabel(
            f'این برنامه از مدل <a href="{BINA_URL}">Bina 0.4 Koochik</a> (ساختهٔ Reza2kn، بر پایهٔ NVIDIA Nemotron-3.5-Parse) '
            f'و نسخهٔ <a href="{ONNX_URL}">ONNX-INT4</a> آن استفاده می‌کند. مدل تحت مجوز OpenMDW-1.1 منتشر شده است '
            '(Copyright © 2026 NVIDIA CORPORATION &amp; AFFILIATES). '
            f'خروجی‌ها ممکن است خطا داشته باشند؛ متن را پیش از استفاده بازبینی کنید. '
            f'<a href="{REPO_URL}">کد برنامه</a> · <a href="{SITE_URL}">نسخهٔ وب</a> · v{__version__}')
        foot.setWordWrap(True)
        foot.setOpenExternalLinks(True)
        foot.setObjectName('muted')
        v.addWidget(foot)
        self.update_buttons()

    # -- model ------------------------------------------------------------
    def refresh_model_box(self):
        ready = downloads.model_ready()
        nvidia = downloads.has_nvidia_gpu()
        cuda = downloads.cuda_ready()
        lines = []
        if not ready:
            lines.append(f'مدل هنوز دانلود نشده است. یک بار حدود <b>{fmt_mb(downloads.MODEL_BYTES)}</b> از Hugging Face '
                         'دانلود می‌شود و روی همین کامپیوتر می‌ماند.')
        elif self.device == 'cuda':
            lines.append('✅ مدل آماده است و با <b>کارت گرافیک NVIDIA (CUDA)</b> اجرا می‌شود.')
        elif self.device == 'cpu':
            lines.append('✅ مدل آماده است و با <b>CPU</b> اجرا می‌شود.')
        else:
            lines.append('مدل دانلود شده است.')
        if nvidia and not cuda:
            lines.append(f'<span style="color:#888">اجرا روی کارت گرافیک NVIDIA آزمایشی است و فعلاً روی بعضی لپ‌تاپ‌ها از CPU کندتر است. '
                         f'برای امتحانش «دانلود CUDA» را بزنید (یک بار حدود {fmt_mb(downloads.CUDA_BYTES)} از بسته‌های رسمی NVIDIA) '
                         'و بعد «کارت گرافیک NVIDIA» را انتخاب کنید.</span>')
        if not nvidia and ready:
            lines.append('<span style="color:#888">کارت گرافیک NVIDIA پیدا نشد؛ برنامه با CPU کار می‌کند.</span>')
        self.model_label.setText('<br>'.join(lines))
        self.btn_model.setVisible(not ready)
        self.btn_cuda.setVisible(nvidia and not cuda)
        self.update_buttons()

    def update_buttons(self):
        downloading = self.busy in ('download-model', 'download-cuda')
        self.btn_model.setEnabled(self.busy is None)
        self.btn_cuda.setEnabled(self.busy is None)
        self.btn_cancel_dl.setVisible(downloading)
        self.device_box.setEnabled(self.busy is None)
        self.btn_start.setEnabled(self.busy is None and bool(self.paths) and downloads.model_ready())
        self.btn_stop.setEnabled(self.busy == 'run')

    def start_download(self, what):
        self.busy = 'download-' + what
        self.dl_bar.setVisible(True)
        self.dl_bar.setValue(0)
        self.set_status('در حال دانلود مدل…' if what == 'model' else 'در حال دانلود کتابخانه‌های CUDA…')
        self.update_buttons()
        self.request_download.emit(what)

    @Slot(int, int)
    def on_download_progress(self, done, total):
        self.dl_bar.setRange(0, max(1, total))
        self.dl_bar.setValue(done)
        self.dl_bar.setFormat(f'{fa(int(done * 100 / max(1, total)))}٪ — {fmt_mb(done * 1024)} از {fmt_mb(total * 1024)}')

    @Slot(str)
    def on_download_done(self, what):
        self.busy = None
        self.dl_bar.setVisible(False)
        self.set_status('دانلود تمام شد.')
        if what == 'cuda':
            self.worker.engine = None  # reload with the GPU
        self.refresh_model_box()
        self.load_engine()

    def on_device_changed(self):
        self.settings.setValue('device-v2', self.device_box.currentData())
        if self.device_box.currentData() == 'cuda' and not downloads.cuda_ready():
            self.set_status('برای کارت گرافیک اول «دانلود CUDA» را بزنید؛ تا آن موقع برنامه با CPU کار می‌کند.')
        if downloads.model_ready():
            self.load_engine()

    def load_engine(self):
        self.busy = 'engine'
        self.update_buttons()
        self.run_bar.setVisible(True)
        self.run_bar.setRange(0, 0)
        choice = self.device_box.currentData()
        # GPU choice: try CUDA, fall back to the CPU if it can't start
        self.request_engine.emit('auto' if choice == 'cuda' and downloads.cuda_ready() else 'cpu')

    @Slot(str, str)
    def on_model_loaded(self, device, reason):
        self.busy = None
        self.device = device
        self.run_bar.setVisible(False)
        self.run_bar.setRange(0, 1000)
        msg = 'مدل آماده است.'
        if reason and self.device_box.currentData() == 'cuda' and downloads.cuda_ready():
            msg += ' (کارت گرافیک راه نیفتاد و برنامه با CPU کار می‌کند: ' + reason[:300] + ')'
        self.set_status(msg)
        self.refresh_model_box()

    @Slot(str, str)
    def on_failed(self, where, message):
        if where in ('model', 'cuda'):
            self.dl_bar.setVisible(False)
        if where == 'engine':
            self.run_bar.setVisible(False)
            self.run_bar.setRange(0, 1000)
        self.busy = None
        self.set_status(message, error=True)
        self.update_buttons()

    def set_status(self, text, error=False):
        self.status_label.setText(text)
        self.status_label.setStyleSheet('color:#c2410c' if error else '')

    # -- files ------------------------------------------------------------
    def pick_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'انتخاب فایل', str(pathlib.Path.home()),
                                                'تصویر و PDF (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff *.pdf)')
        self.add_files(paths)

    @Slot(list)
    def add_files(self, paths):
        for p in paths:
            p = str(pathlib.Path(p))
            if is_supported(p) and p not in self.paths:
                self.paths.append(p)
                try:
                    doc = Document(p)
                    meta = f'{fa(doc.page_count)} صفحه' if doc.is_pdf else 'تصویر'
                except Exception:
                    meta = 'خراب؟'
                item = QListWidgetItem(f'{pathlib.Path(p).name}   ({meta})')
                item.setData(Qt.UserRole, p)
                self.file_list.addItem(item)
        self.update_buttons()

    def remove_selected(self):
        if self.busy == 'run':
            return
        for item in self.file_list.selectedItems():
            self.paths.remove(item.data(Qt.UserRole))
            self.file_list.takeItem(self.file_list.row(item))
        self.update_buttons()

    # -- running ----------------------------------------------------------
    def start_run(self):
        if not self.paths or self.worker.engine is None:
            if self.worker.engine is None:
                self.set_status('مدل هنوز بالا نیامده است.', error=True)
            return
        self.clear_results()
        self.result_docs = []
        for p in self.paths:
            doc = Document(p)
            self.result_docs.append({'name': doc.name, 'is_pdf': doc.is_pdf, 'mode': self.mode_box.currentData(),
                                     'pages': [{'number': n, 'text': ''} for n in doc.page_numbers(self.pages_edit.text())]})
        total_pages = sum(len(d['pages']) for d in self.result_docs)
        device = self.worker.engine.device
        rate = float(self.settings.value(f'rate-{device}', DEFAULT_RATE[device]))
        self.progress = Progress(total_pages, rate)
        self.build_result_editors()
        self.busy = 'run'
        self.update_buttons()
        self.run_bar.setVisible(True)
        self.run_bar.setValue(0)
        self.eta_label.setVisible(True)
        self.eta_label.setText('۰٪ — در حال تخمین زمان…')
        self.run_started = time.monotonic()
        self.request_run.emit(list(self.paths), {'mode': self.mode_box.currentData(), 'pages': self.pages_edit.text(),
                                                 'pdf_width': self.quality_box.currentData()})

    def build_result_editors(self):
        self.results = {}
        for i, doc in enumerate(self.result_docs):
            if len(self.result_docs) > 1:
                h = QLabel(doc['name'])
                h.setObjectName('docname')
                self.results_area.addWidget(h)
            for page in doc['pages']:
                if doc['is_pdf'] or len(doc['pages']) > 1:
                    self.results_area.addWidget(QLabel(f'صفحهٔ {fa(page["number"])}'))
                edit = QPlainTextEdit()
                edit.setMinimumHeight(160)
                edit.setLayoutDirection(Qt.RightToLeft)
                option = edit.document().defaultTextOption()
                option.setTextDirection(Qt.RightToLeft)
                option.setAlignment(Qt.AlignRight)
                edit.document().setDefaultTextOption(option)
                edit.setPlaceholderText('در صف…')
                self.results_area.addWidget(edit)
                self.results[(i, page['number'])] = edit
        self.results_box.setVisible(True)

    @Slot(dict)
    def on_run_progress(self, m):
        p = self.progress
        stage = m['stage']
        if stage == 'page':
            p.add_page(m['patches'])
        elif stage == 'block':
            p.current = 0.05
            p.current_patches = m['patches']
        elif stage == 'block-done':
            p.done += m['patches']
            p.seconds += m['seconds']
            p.current = 0
            p.current_patches = 0
            self.settings.setValue(f"rate-{m['device']}", p.rate)
        if 'index' in m and stage != 'block-done':
            p.current = max(p.current, block_fraction(stage, m.get('tokens', 0)))
        texts = {
            'render': 'آماده کردن صفحه',
            'page': 'تحلیل صفحه',
            'block': 'خواندن متن',
            'vision': 'خواندن متن',
            'decode': 'خواندن متن',
            'retry': 'خواندن دوباره با کیفیت بالاتر',
        }
        if stage in texts:
            parts = [m.get('label', ''), texts[stage]]
            if 'count' in m:
                parts.append(f'بخش {fa(m["index"] + 1)} از {fa(m["count"])}')
            if stage == 'decode':
                parts.append(f'{fa(m["tokens"])} توکن')
            self.set_status(' — '.join(x for x in parts if x))
        self.show_eta()

    def show_eta(self):
        p = self.progress
        if p is None:
            return
        self.run_bar.setValue(int(p.fraction * 1000))
        pct = f'{fa(int(p.fraction * 100))}٪'
        if not p.pages:
            self.eta_label.setText(f'{pct} — در حال تخمین زمان…')
        else:
            prefix = 'حدود' if p.done else 'تخمین اولیه: حدود'
            self.eta_label.setText(f'{pct} — {prefix} {fmt_duration(p.remaining)} مانده')

    @Slot(int, int, str)
    def on_page_done(self, i, number, text):
        edit = self.results.get((i, number))
        if edit is not None:
            edit.setPlainText(text)
            if not text:
                edit.setPlaceholderText('متنی پیدا نشد.')

    @Slot(bool)
    def on_run_finished(self, completed):
        self.busy = None
        self.update_buttons()
        self.run_bar.setVisible(False)
        elapsed = time.monotonic() - self.run_started
        if completed:
            self.eta_label.setText(f'۱۰۰٪ — تمام شد در {fmt_duration(elapsed)}')
            self.set_status('تمام شد. حالا می‌توانید خروجی را ذخیره کنید.')
        else:
            self.eta_label.setVisible(False)
            if not self.status_label.styleSheet():
                self.set_status('متوقف شد. متن صفحه‌های تمام‌شده در بخش نتیجه هست.')

    # -- results ----------------------------------------------------------
    def collect(self):
        docs = []
        for i, doc in enumerate(self.result_docs):
            pages = [{'number': pg['number'], 'text': self.results[(i, pg['number'])].toPlainText()} for pg in doc['pages']]
            docs.append({**doc, 'pages': pages})
        return docs

    def export(self, kind):
        docs = self.collect()
        if not docs:
            return
        name = exporters.base_name(docs) + '.' + kind
        start = str(pathlib.Path(self.paths[0]).parent / name) if self.paths else name
        filters = {'docx': 'Word (*.docx)', 'txt': 'Text (*.txt)', 'md': 'Markdown (*.md)'}
        path, _ = QFileDialog.getSaveFileName(self, 'ذخیرهٔ خروجی', start, filters[kind])
        if not path:
            return
        try:
            if kind == 'docx':
                exporters.to_docx(docs, path)
            else:
                text = exporters.to_text(docs) if kind == 'txt' else exporters.to_markdown(docs)
                pathlib.Path(path).write_text(text, encoding='utf-8')
            self.set_status(f'ذخیره شد: {path}')
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(pathlib.Path(path).parent)))
        except Exception as error:
            self.set_status(f'ذخیره نشد: {error}', error=True)

    def copy_all(self):
        docs = self.collect()
        if docs:
            QGuiApplication.clipboard().setText(exporters.to_text(docs))
            self.set_status('همهٔ متن کپی شد.')

    def clear_results(self):
        while self.results_area.count():
            w = self.results_area.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.results = {}
        self.result_docs = []
        self.results_box.setVisible(False)
        if self.busy is None:
            self.eta_label.setVisible(False)

    def closeEvent(self, event):
        self.worker.cancel.set()
        self.thread.quit()
        self.thread.wait(3000)
        super().closeEvent(event)


STYLE = """
* { font-size: 10.5pt; }
QMainWindow, QScrollArea, QScrollArea > QWidget > QWidget { background: %(bg)s; }
QLabel#title { font-size: 20pt; font-weight: 700; }
QLabel#muted { color: %(muted)s; }
QLabel#eta { font-weight: 600; color: %(accent)s; }
QLabel#docname { font-weight: 700; margin-top: 8px; }
QGroupBox { background: %(surface)s; border: 1px solid %(line)s; border-radius: 12px; margin-top: 14px; padding: 14px 12px 10px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top right; right: 14px; padding: 0 6px; color: %(accent)s; }
QPushButton { padding: 6px 14px; border-radius: 8px; border: 1px solid %(line)s; background: %(surface)s; }
QPushButton:hover { border-color: %(accent)s; }
QPushButton:disabled { color: %(muted)s; }
QPushButton#primary { background: %(accent)s; color: white; border-color: %(accent)s; font-weight: 600; }
QPushButton#primary:disabled { background: %(line)s; border-color: %(line)s; }
QProgressBar { border: 1px solid %(line)s; border-radius: 6px; background: %(bg)s; height: 16px; text-align: center; }
QProgressBar::chunk { background: %(accent)s; border-radius: 6px; }
QPlainTextEdit, QListWidget, QLineEdit, QComboBox { border: 1px solid %(line)s; border-radius: 8px; background: %(bg)s; padding: 4px; }
"""


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    app.setApplicationName('FarsiOCR')
    app.setLayoutDirection(Qt.RightToLeft)
    font_path = downloads.resource_path('assets', 'Vazirmatn-Regular.ttf')
    family = None
    if font_path.is_file():
        fid = QFontDatabase.addApplicationFont(str(font_path))
        bold = downloads.resource_path('assets', 'Vazirmatn-Bold.ttf')
        if bold.is_file():
            QFontDatabase.addApplicationFont(str(bold))
        families = QFontDatabase.applicationFontFamilies(fid)
        family = families[0] if families else None
    app.setFont(QFont(family or 'Tahoma', 10))
    icon = downloads.resource_path('assets', 'icon.ico')
    if icon.is_file():
        app.setWindowIcon(QIcon(str(icon)))
    dark = app.styleHints().colorScheme() == Qt.ColorScheme.Dark
    colors = ({'bg': '#18181b', 'surface': '#232327', 'line': '#3a3a40', 'muted': '#a1a1aa', 'accent': '#fb923c'} if dark else
              {'bg': '#f4f4f5', 'surface': '#ffffff', 'line': '#dcdde1', 'muted': '#62646b', 'accent': '#ea580c'})
    app.setStyleSheet(STYLE % colors)
    window = MainWindow()
    window.show()
    return app.exec()
