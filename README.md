# OCR فارسی (تحت وب، با مدل بینا)

برنامهٔ وب برای تبدیل **عکس** و **PDF** فارسی به متن، با خروجی **TXT**، **Word (.docx)** و **Markdown (.md)**.

تشخیص متن با مدل **[Bina 0.4 Koochik](https://huggingface.co/Reza2kn/Bina-0.4-Koochik)** ساختهٔ [Reza2kn](https://huggingface.co/Reza2kn) انجام می‌شود،
با نسخهٔ مرورگری آن: [`Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4`](https://huggingface.co/Reza2kn/Bina-0.4-Koochik-ONNX-WebGPU-INT4).

## چطور کار می‌کند؟

- **سرور لازم نیست.** مدل با WebGPU روی کارت گرافیک خود کاربر و داخل مرورگر اجرا می‌شود. فایل‌ها هیچ‌وقت از کامپیوتر کاربر بیرون نمی‌روند.
  به همین دلیل سایت کاملاً استاتیک است و روی Cloudflare Pages (رایگان) میزبانی می‌شود.
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

## استقرار روی Cloudflare Pages

### روش ۱: اتصال به GitHub (پیشنهادی)

1. در داشبورد Cloudflare به **Workers & Pages → Create → Pages → Connect to Git** بروید و همین مخزن را انتخاب کنید.
2. تنظیمات build:
   - **Framework preset:** `None`
   - **Build command:** `npm run build`
   - **Build output directory:** `dist`
   - **Environment variable** (اختیاری): `NODE_VERSION` = `20`
3. **Save and Deploy**. بعد از چند دقیقه سایت روی `https://<نام-پروژه>.pages.dev` بالا می‌آید و با هر push به‌روز می‌شود.
   دامنهٔ شخصی را از **Custom domains** اضافه کنید.

### روش ۲: از خط فرمان با Wrangler

```bash
npm install
npm run build
npx wrangler pages deploy dist --project-name farsi-ocr
```

### نکته‌ها

- Cloudflare Pages حداکثر ۲۵ مگابایت برای هر فایل اجازه می‌دهد؛ برای همین فایل `.wasm` اجرای ONNX (۲۹٫۵ مگابایت) و وزن‌های مدل از Hugging Face بارگذاری می‌شوند و در build فقط مسیر آن در worker عوض می‌شود.
- سایت باید روی **HTTPS** باشد (pages.dev خودش HTTPS است). WebGPU روی HTTP کار نمی‌کند.
- فایل `src/_headers` هدرهای کش را برای Cloudflare تنظیم می‌کند.

## ساختار

```
src/                 کد برنامه (HTML/CSS/JS بدون فریم‌ورک)
  app.js             رابط کاربری، خواندن PDF/عکس، صف پردازش
  bina-client.js     ارتباط با worker مدل و worker چیدمان
  exporters.js       ساخت TXT / Markdown / Word
  config.js          revision ثابت مدل
scripts/build.mjs    ساخت dist/ (کپی pdf.js و docx، دانلود و وصلهٔ فایل‌های اجرای بینا)
scripts/worker-prelude.js  کش‌کردن فایل‌های مدل و گزارش پیشرفت دانلود داخل worker
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
