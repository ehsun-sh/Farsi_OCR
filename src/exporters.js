// Turns OCR results into .txt, .md and .docx files.
//
// A result looks like:
//   [{name: 'scan.pdf', mode: 'text' | 'markdown', pages: [{number: 1, text: '...'}]}]

const RTL_CHARS = /[֐-ࣿיִ-﷿ﹰ-﻿]/;
const isRTL = (s) => RTL_CHARS.test(s);
const faDigits = (n) => String(n).replace(/\d/g, (d) => '۰۱۲۳۴۵۶۷۸۹'[d]);

function pageLabel(doc, page) {
  return doc.pages.length > 1 || doc.isPdf ? `صفحهٔ ${faDigits(page.number)}` : null;
}

export function baseName(docs) {
  if (docs.length === 1) return docs[0].name.replace(/\.[^.]+$/, '') + '-ocr';
  return 'farsi-ocr';
}

export function toText(docs) {
  const multi = docs.length > 1;
  const parts = [];
  for (const doc of docs) {
    const out = [];
    if (multi) out.push(`===== ${doc.name} =====`);
    for (const page of doc.pages) {
      const label = pageLabel(doc, page);
      if (label) out.push(`--- ${label} ---`);
      out.push(page.text.trim());
      out.push('');
    }
    parts.push(out.join('\n').trim());
  }
  return parts.join('\n\n') + '\n';
}

export function toMarkdown(docs) {
  const multi = docs.length > 1;
  const parts = [];
  for (const doc of docs) {
    const out = [];
    if (multi) out.push(`# ${doc.name}`, '');
    for (const page of doc.pages) {
      const label = pageLabel(doc, page);
      if (label) out.push(`${multi ? '##' : '#'} ${label}`, '');
      const text = page.text.trim();
      if (doc.mode === 'markdown') out.push(text);
      // Plain OCR text: keep every line break as a Markdown hard break.
      else out.push(text.split('\n').map((l) => l.trimEnd()).join('  \n').replace(/  \n(\s*  \n)+/g, '\n\n'));
      out.push('');
    }
    parts.push(out.join('\n').trim());
  }
  return parts.join('\n\n') + '\n';
}

// --- Word ---------------------------------------------------------------

const FONT = {ascii: 'Calibri', hAnsi: 'Calibri', cs: 'Tahoma'};

function runsFromMarkdownInline(docx, text, rtl, extra = {}) {
  // Supports **bold** and *italic*; everything else is literal text.
  const runs = [];
  const re = /(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) runs.push(new docx.TextRun({text: text.slice(last, m.index), rightToLeft: rtl, font: FONT, ...extra}));
    const bold = m[0].startsWith('**');
    const inner = bold ? m[0].slice(2, -2) : m[0].slice(1, -1);
    runs.push(new docx.TextRun({text: inner, bold, italics: !bold, rightToLeft: rtl, font: FONT, ...extra}));
    last = m.index + m[0].length;
  }
  if (last < text.length) runs.push(new docx.TextRun({text: text.slice(last), rightToLeft: rtl, font: FONT, ...extra}));
  return runs;
}

function paragraph(docx, line, markdown) {
  const rtl = isRTL(line);
  const base = {bidirectional: rtl, alignment: rtl ? docx.AlignmentType.RIGHT : docx.AlignmentType.LEFT};
  if (markdown) {
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const level = [docx.HeadingLevel.HEADING_2, docx.HeadingLevel.HEADING_3, docx.HeadingLevel.HEADING_4][Math.min(h[1].length, 3) - 1];
      return new docx.Paragraph({...base, heading: level, children: runsFromMarkdownInline(docx, h[2], rtl)});
    }
    const li = /^\s*(?:[-*+]|\d+[.)])\s+(.*)$/.exec(line);
    if (li) return new docx.Paragraph({...base, bullet: {level: 0}, children: runsFromMarkdownInline(docx, li[1], rtl)});
    return new docx.Paragraph({...base, children: runsFromMarkdownInline(docx, line, rtl)});
  }
  return new docx.Paragraph({...base, children: [new docx.TextRun({text: line, rightToLeft: rtl, font: FONT})]});
}

export async function toDocx(docs) {
  const docx = window.docx;
  if (!docx) throw new Error('docx library not loaded');
  const sections = docs.map((doc, docIndex) => {
    const children = [];
    if (docs.length > 1) {
      children.push(new docx.Paragraph({
        heading: docx.HeadingLevel.TITLE, bidirectional: isRTL(doc.name), alignment: docx.AlignmentType.CENTER,
        children: [new docx.TextRun({text: doc.name, font: FONT, rightToLeft: isRTL(doc.name)})],
      }));
    }
    doc.pages.forEach((page, i) => {
      const label = pageLabel(doc, page);
      if (label) {
        children.push(new docx.Paragraph({
          heading: docx.HeadingLevel.HEADING_1, bidirectional: true, alignment: docx.AlignmentType.RIGHT,
          pageBreakBefore: i > 0,
          children: [new docx.TextRun({text: label, rightToLeft: true, font: FONT})],
        }));
      }
      const lines = page.text.trim().split('\n');
      for (const line of lines) {
        if (doc.mode === 'markdown' && /^\s*\|?\s*:?-{3,}/.test(line)) continue; // table separator rows
        children.push(paragraph(docx, line.replace(/\s+$/, ''), doc.mode === 'markdown'));
      }
    });
    return {
      properties: docIndex > 0 ? {type: docx.SectionType.NEXT_PAGE} : {},
      children,
    };
  });
  const document = new docx.Document({
    creator: 'Farsi OCR (Bina 0.4 Koochik)',
    title: baseName(docs),
    description: 'Text recognised with the Bina 0.4 Koochik OCR model',
    styles: {
      default: {document: {run: {font: FONT, size: 24, rightToLeft: true}, paragraph: {spacing: {after: 80}}}},
    },
    sections,
  });
  return docx.Packer.toBlob(document);
}
