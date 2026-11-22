"""Writes OCR results as .txt, .md and .docx (same layout as the web app).

docs: [{'name': 'scan.pdf', 'mode': 'text'|'markdown', 'is_pdf': bool,
        'pages': [{'number': 1, 'text': '...'}]}]
"""
import re

RTL_CHARS = re.compile('[֐-ࣿיִ-﷿ﹰ-﻿]')
FA_DIGITS = str.maketrans('0123456789', '۰۱۲۳۴۵۶۷۸۹')


def is_rtl(s):
    return bool(RTL_CHARS.search(s))


def page_label(doc, page):
    if len(doc['pages']) > 1 or doc.get('is_pdf'):
        return 'صفحهٔ ' + str(page['number']).translate(FA_DIGITS)
    return None


def base_name(docs):
    if len(docs) == 1:
        return re.sub(r'\.[^.]+$', '', docs[0]['name']) + '-ocr'
    return 'farsi-ocr'


def to_text(docs):
    multi = len(docs) > 1
    parts = []
    for doc in docs:
        out = []
        if multi:
            out.append(f"===== {doc['name']} =====")
        for page in doc['pages']:
            label = page_label(doc, page)
            if label:
                out.append(f'--- {label} ---')
            out.append(page['text'].strip())
            out.append('')
        parts.append('\n'.join(out).strip())
    return '\n\n'.join(parts) + '\n'


def to_markdown(docs):
    multi = len(docs) > 1
    parts = []
    for doc in docs:
        out = []
        if multi:
            out += [f"# {doc['name']}", '']
        for page in doc['pages']:
            label = page_label(doc, page)
            if label:
                out += [f"{'##' if multi else '#'} {label}", '']
            text = page['text'].strip()
            if doc.get('mode') == 'markdown':
                out.append(text)
            else:  # plain OCR text: keep each line break as a Markdown hard break
                joined = '  \n'.join(line.rstrip() for line in text.split('\n'))
                out.append(re.sub(r'  \n(\s*  \n)+', '\n\n', joined))
            out.append('')
        parts.append('\n'.join(out).strip())
    return '\n\n'.join(parts) + '\n'


# --- Word ---------------------------------------------------------------

def _set_rtl(paragraph, rtl):
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    ppr = paragraph._p.get_or_add_pPr()
    if rtl:
        bidi = OxmlElement('w:bidi')
        bidi.set(qn('w:val'), '1')
        ppr.insert(0, bidi)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.RIGHT if rtl else WD_ALIGN_PARAGRAPH.LEFT


def _add_run(paragraph, text, rtl, bold=False, italic=False):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    run = paragraph.add_run(text)
    run.bold = bold or None
    run.italic = italic or None
    rpr = run._r.get_or_add_rPr()
    fonts = rpr.find(qn('w:rFonts'))
    if fonts is None:
        fonts = OxmlElement('w:rFonts')
        rpr.insert(0, fonts)
    for key, value in (('w:ascii', 'Calibri'), ('w:hAnsi', 'Calibri'), ('w:cs', 'Tahoma')):
        fonts.set(qn(key), value)
    if rtl:
        rpr.append(OxmlElement('w:rtl'))
    return run


INLINE = re.compile(r'(\*\*[^*]+\*\*|\*[^*\s][^*]*\*)')


def _add_inline(paragraph, text, rtl):
    last = 0
    for m in INLINE.finditer(text):
        if m.start() > last:
            _add_run(paragraph, text[last:m.start()], rtl)
        bold = m.group(0).startswith('**')
        _add_run(paragraph, m.group(0)[2:-2] if bold else m.group(0)[1:-1], rtl, bold=bold, italic=not bold)
        last = m.end()
    if last < len(text):
        _add_run(paragraph, text[last:], rtl)


def to_docx(docs, path):
    from docx import Document
    from docx.enum.section import WD_SECTION
    from docx.oxml.ns import qn
    from docx.shared import Pt

    document = Document()
    style = document.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(12)
    style.element.rPr.rFonts.set(qn('w:cs'), 'Tahoma')
    style.paragraph_format.space_after = Pt(4)
    document.core_properties.author = 'Farsi OCR (Bina 0.4 Koochik)'
    document.core_properties.title = base_name(docs)
    document.core_properties.comments = 'Text recognised with the Bina 0.4 Koochik OCR model'

    for d, doc in enumerate(docs):
        if d > 0:
            document.add_section(WD_SECTION.NEW_PAGE)
        if len(docs) > 1:
            p = document.add_paragraph(style='Title')
            _set_rtl(p, is_rtl(doc['name']))
            _add_run(p, doc['name'], is_rtl(doc['name']))
        markdown = doc.get('mode') == 'markdown'
        for i, page in enumerate(doc['pages']):
            label = page_label(doc, page)
            if label:
                p = document.add_paragraph(style='Heading 1')
                p.paragraph_format.page_break_before = i > 0
                _set_rtl(p, True)
                _add_run(p, label, True)
            for line in page['text'].strip().split('\n'):
                line = line.rstrip()
                if markdown and re.match(r'^\s*\|?\s*:?-{3,}', line):
                    continue  # table separator rows
                rtl = is_rtl(line)
                if markdown:
                    h = re.match(r'^(#{1,6})\s+(.*)$', line)
                    li = re.match(r'^\s*(?:[-*+]|\d+[.)])\s+(.*)$', line)
                    if h:
                        p = document.add_paragraph(style=f'Heading {min(len(h.group(1)), 3) + 1}')
                        _set_rtl(p, rtl)
                        _add_inline(p, h.group(2), rtl)
                    elif li:
                        p = document.add_paragraph(style='List Bullet')
                        _set_rtl(p, rtl)
                        _add_inline(p, li.group(1), rtl)
                    else:
                        p = document.add_paragraph()
                        _set_rtl(p, rtl)
                        _add_inline(p, line, rtl)
                else:
                    p = document.add_paragraph()
                    _set_rtl(p, rtl)
                    _add_run(p, line, rtl)
    document.save(str(path))
