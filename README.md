# OCR فارسی (تحت وب، با مدل بینا)

برنامهٔ وب برای تبدیل **عکس** و **PDF** فارسی به متن، با خروجی **TXT**، **Word (.docx)** و **Markdown (.md)**.

تشخیص متن با مدل **[Bina 0.4 Koochik](https://huggingface.co/Reza2kn/Bina-0.4-Koochik)** ساختهٔ [Reza2kn](https://huggingface.co/Reza2kn) انجام می‌شود،
با نسخهٔ مرورگری آن: [`Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4`](https://huggingface.co/Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4).

## چطور کار می‌کند؟

- **سرور لازم نیست.** مدل با WebGPU روی کارت گرافیک خود کاربر و داخل مرورگر اجرا می‌شود. فایل‌ها هیچ‌وقت از کامپیوتر کاربر بیرون نمی‌روند.
  به همین دلیل سایت استاتیک است و روی Cloudflare (رایگان) میزبانی می‌شود؛ فقط یک Worker کوچک فایل‌های مدل را از Hugging Face رد می‌کند.
- بار اول حدود **۷۷۰ مگابایت** فایل مدل مستقیماً از Hugging Face دانلود و در Cache Storage مرورگر ذخیره می‌شود؛ دفعه‌های بعد از همان‌جا خوانده می‌شود.
- PDF با [pdf.js](https://mozilla.github.io/pdf.js/) صفحه‌به‌صفحه به تصویر تبدیل می‌شود، هر صفحه با کد چیدمان خود بینا به ستون‌ها و بلوک‌ها تقسیم می‌شود و بلوک‌ها یکی‌یکی به مدل داده می‌شوند (با نمایش پیشرفت و امکان توقف).
- خروجی Word راست‌به‌چپ است و برای هر صفحهٔ PDF یک عنوان و شکستن صفحه دارد.
- دو حالت خروجی مدل: «متن ساده» (پیشنهادی، همان حالتی که مدل با آن ارزیابی شده) و «Markdown» آزمایشی برای عنوان‌ها و جدول‌ها.

### پیش‌نیاز کاربر

نسخهٔ جدید **Chrome** یا **Edge** روی کامپیوتر (ویندوز/مک/لینوکس) با WebGPU فعال. اگر WebGPU در دسترس نباشد، برنامه پیام روشنی نشان می‌دهد.
اجرای بدون کارت گرافیک (WASM) پشتیبانی نمی‌شود، چون نسخهٔ مرورگری بینا فقط برای WebGPU ساخته و آزموده شده (کرنل سفارشی ONNX Runtime برای دقت BF16).
روی موبایل به‌خاطر حافظهٔ گرافیکی ممکن است کار نکند.

## اجرای محلی

```bash
npm install
npm run dev          # build + سرور محلی روی http://127.0.0.1:8788
```

`npm run build` خروجی را در `dist/` می‌سازد. این دستور چند فایل کوچک اجرای بینا (worker، ONNX Runtime سفارشی، کد چیدمان و مجوزها) را از Hugging Face،
از یک revision ثابت، دانلود و با SHA-256 بررسی می‌کند. فایل‌های بزرگ مدل در build نیستند و مرورگر کاربر مستقیماً از Hugging Face می‌گیرد.

## برنامهٔ کامپیوتر (Windows، macOS، Linux)

اگر نسخهٔ وب برایتان کند است، نسخهٔ ویندوز همین مدل را مستقیم با **ONNX Runtime** روی CPU (و به‌صورت آزمایشی روی کارت گرافیک NVIDIA با CUDA) اجرا می‌کند. روی یک لپ‌تاپ با RTX 5070، سه خط متن با CPU در حدود ۷ ثانیه خوانده شد.

- دانلود از [Releases](https://github.com/ehsun-sh/Farsi_OCR/releases/latest)؛ نصب لازم ندارد:
  - Windows 10/11: `FarsiOCR-Windows.zip` را باز کنید و `FarsiOCR.exe` را اجرا کنید.
  - macOS 14+ (Apple Silicon): `FarsiOCR-macOS.zip` را باز کنید و `FarsiOCR.app` را بار اول با راست‌کلیک ← Open باز کنید (برنامه امضای اپل ندارد).
  - Linux x86-64: `tar -xzf FarsiOCR-Linux.tar.gz && ./FarsiOCR/FarsiOCR`
- بار اول، مدل [Bina-0.4-Koochik-ONNX-INT4](https://huggingface.co/Reza2kn/Bina-0.4-Koochik-ONNX-INT4) (حدود ۷۴۰ مگابایت، نسخهٔ ثابت و با بررسی هش) از Hugging Face در `%LOCALAPPDATA%\FarsiOCR` (ویندوز)، `~/Library/Application Support/FarsiOCR` (مک) یا `~/.local/share/FarsiOCR` (لینوکس) دانلود می‌شود.
- (فقط ویندوز) اجرا روی کارت NVIDIA فعلاً آزمایشی است و در تست ما از CPU کندتر بود. دکمهٔ «دانلود CUDA» کتابخانه‌های CUDA 13 و cuDNN 9 (حدود ۱٫۱ گیگابایت) را یک بار از بسته‌های رسمی NVIDIA در PyPI می‌گیرد. درایور NVIDIA باید به‌روز باشد (سری ۵۸۰ یا جدیدتر). اگر کارت گرافیک راه نیفتد، برنامه خودکار با CPU کار می‌کند.
- ورودی: عکس (PNG، JPG، WebP، TIFF…) و PDF. خروجی: Word، TXT و Markdown؛ مثل نسخهٔ وب.
- حالت خط فرمان هم دارد: `FarsiOCR.exe --ocr scan.pdf --out scan.docx --device cpu --log run.log`

کد در پوشهٔ `desktop/` است (Python + PySide6). پیش‌پردازش تصویر (`preprocess.py`) پورت دقیق `preprocess.js` مدل است و خروجی‌اش با نسخهٔ وب بیت‌به‌بیت یکی است. هر سه نسخه را GitHub Actions (`.github/workflows/desktop.yml`) با PyInstaller می‌سازد و در Releases می‌گذارد. اجرای از روی کد:

```bash
cd desktop
pip install -r requirements.txt
python main.py
```

## استقرار روی Cloudflare

> **چرا یک Worker کوچک هم لازم است؟** Hugging Face به بعضی دامنه‌های میزبانی (مثل `*.workers.dev`) اجازهٔ دانلود مستقیم از مرورگر (CORS) نمی‌دهد.
> برای همین `worker/hf-proxy.js` فایل‌های مدل را سمت Cloudflare از Hugging Face می‌گیرد و از همان دامنهٔ سایت زیر مسیر `/hf/` تحویل می‌دهد
> (فقط همان نسخهٔ ثابت مدل؛ پراکسی باز نیست). Cloudflare این فایل‌ها را کش می‌کند. اگر پراکسی در دسترس نباشد، برنامه مستقیماً از Hugging Face دانلود می‌کند.

### روش ۱: Cloudflare Workers (پیشنهادی)

1. در داشبورد Cloudflare به **Workers & Pages → Create → Import a repository** بروید و این مخزن را انتخاب کنید.
2. تنظیمات:
   - **Project name:** `farsi-ocr` (باید با `name` در `wrangler.jsonc` یکی باشد)
   - **Build command:** `npm run build`
   - **Deploy command:** `npx wrangler deploy`
3. **Deploy**. سایت روی `https://farsi-ocr.<حساب-شما>.workers.dev` بالا می‌آید و با هر push به `main` به‌روز می‌شود.
   دامنهٔ شخصی را از **Settings → Domains & Routes** اضافه کنید.

از خط فرمان: `npm install && npm run build && npx wrangler deploy`

### روش ۲: Cloudflare Pages

**Workers & Pages → Create → Pages → Connect to Git**، با Build command `npm run build`، Build output directory `dist` و متغیر `NODE_VERSION=20`.
پوشهٔ `functions/` همان پراکسی `/hf/` را به‌صورت Pages Function فعال می‌کند.

### نکته‌ها

- Cloudflare حداکثر ۲۵ مگابایت برای هر فایل استاتیک اجازه می‌دهد؛ برای همین فایل `.wasm` اجرای ONNX (۲۹٫۵ مگابایت) و وزن‌های مدل از طریق `/hf/` (یا مستقیم از Hugging Face) بارگذاری می‌شوند.
- سایت باید روی **HTTPS** باشد. WebGPU روی HTTP کار نمی‌کند.
- فایل `src/_headers` هدرهای کش فایل‌های استاتیک را تنظیم می‌کند.

## ساختار

```
src/                 کد برنامه (HTML/CSS/JS بدون فریم‌ورک)
  app.js             رابط کاربری، خواندن PDF/عکس، صف پردازش
  bina-client.js     ارتباط با worker مدل و worker چیدمان
  exporters.js       ساخت TXT / Markdown / Word
  config.js          revision ثابت مدل
scripts/build.mjs    ساخت dist/ (کپی pdf.js و docx، دانلود و وصلهٔ فایل‌های اجرای بینا)
scripts/worker-prelude.js  کش‌کردن فایل‌های مدل و گزارش پیشرفت دانلود داخل worker
worker/              Cloudflare Worker: سرو dist/ و پراکسی /hf/ برای فایل‌های مدل
functions/hf/        همان پراکسی برای Cloudflare Pages
wrangler.jsonc       تنظیمات Worker
```

## مجوز و اعتبار (Credits & License)

- **مدل:** [Bina 0.4 Koochik](https://huggingface.co/Reza2kn/Bina-0.4-Koochik) از Reza2kn، بر پایهٔ NVIDIA Nemotron-3.5-Parse؛
  نسخهٔ مرورگری [Bina-0.4-Koochik-ONNX-WebGPU-INT4](https://huggingface.co/Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4) (checkpoint 18,404).
  وزن‌ها و کد اجرای مدل تحت **OpenMDW-1.1** منتشر شده‌اند — Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
  OpenMDW-1.1 استفاده، تغییر و توزیع (حتی تجاری) را آزاد می‌گذارد، به این شرط که در هر توزیع متن مجوز و اطلاعیه‌های حق نشر حفظ شوند؛
  هیچ محدودیتی روی خروجی‌های مدل (متن‌های OCR شده) نمی‌گذارد. این پروژه متن مجوز را در `dist/bina/LICENSE` منتشر می‌کند و در پانویس سایت به آن لینک می‌دهد.
- **ONNX Runtime Web** (سفارشی‌شدهٔ بینا، MIT)، **@huggingface/tokenizers** (Apache-2.0)، **fast-png** (MIT): مجوزها همراه فایل‌های اجرا در `dist/bina/` منتشر می‌شوند.
- **pdf.js** (Apache-2.0) و **docx** (MIT): مجوزها در `dist/vendor/`.
- دقت گزارش‌شدهٔ مدل مربوط به مجموعهٔ ارزیابی خود سازنده است و برای هر ورودی تضمین نمی‌شود؛ متن خروجی را پیش از استفاده بازبینی کنید.
