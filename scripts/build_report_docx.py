"""Создаёт Word-версию отчёта из проверенного Markdown."""
from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
root=Path(__file__).resolve().parents[1]
lines=(root/'docs/REPORT.md').read_text().splitlines()
doc=Document()
sec=doc.sections[0]
sec.top_margin=sec.bottom_margin=Inches(.7)
sec.left_margin=sec.right_margin=Inches(.75)
normal=doc.styles['Normal']
normal.font.name='Arial';normal.font.size=Pt(10)
normal.paragraph_format.space_after=Pt(6)
for name in ['Title','Heading 1','Heading 2']:
    doc.styles[name].font.name='Arial'
    doc.styles[name].font.color.rgb=RGBColor(0,0,0)
doc.styles['Title'].font.size=Pt(24)
doc.styles['Heading 1'].font.size=Pt(15)
def plain(s):
    s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'\1 (\2)',s)
    return s.replace('`','').replace('**','')
i=0
while i<len(lines):
    line=lines[i]
    if not line.strip():
        i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].startswith('|'):
            cells=[plain(s.strip()) for s in lines[i].strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',s) for s in cells):rows.append(cells)
            i+=1
        t=doc.add_table(rows=1, cols=len(rows[0]));t.style='Light Shading Accent 1'
        for j,s in enumerate(rows[0]):t.rows[0].cells[j].text=s
        rep=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(rep)
        for row in rows[1:]:
            cells=t.add_row().cells
            for j,s in enumerate(row):cells[j].text=s
        for row in t.rows:
            prevent=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(prevent)
            for cell in row.cells:
                for p in cell.paragraphs:
                    for run in p.runs:run.font.size=Pt(8 if len(rows[0])>=6 else 9)
        for row_index, row in enumerate(t.rows):
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = row_index < len(t.rows)-1
        doc.add_paragraph()
        continue
    if line.startswith('# '):
        doc.add_paragraph(plain(line[2:]),'Title');i+=1;continue
    if line.startswith('## '):
        heading=line[3:]
        doc.add_heading(plain(heading),level=1);i+=1;continue
    if line.startswith('- '):
        doc.add_paragraph(plain(line[2:]),style='List Bullet');i+=1;continue
    para=[line];i+=1
    while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','- ')):
        para.append(lines[i]);i+=1
    doc.add_paragraph(plain(' '.join(para)))
doc.add_heading('Графики данных',level=1)
for name,caption in [('severity','Рисунок 1. Распределение Severity по полному набору'),('hour','Рисунок 2. Зарегистрированные ДТП по местному часу начала'),('missing','Рисунок 3. Столбцы с наибольшим числом пропусков')]:
    doc.add_picture(str(root/f'outputs/figures/{name}.png'),width=Inches(6.4))
    doc.add_paragraph(caption)
doc.core_properties.title='US Traffic Accidents Patterns and Severity Prediction'
doc.core_properties.author='Bekmukhamedova Meruert; Aiganym Gadilova'
# Удаляем унаследованную синюю линию из стиля Title.
for style in doc.styles:
    for border in list(style.element.iter(qn('w:pBdr'))):
        border.getparent().remove(border)
for border in list(doc.element.iter(qn('w:pBdr'))):
    border.getparent().remove(border)
doc.save(root/'deliverables/US_Traffic_Accidents_Report.docx')
