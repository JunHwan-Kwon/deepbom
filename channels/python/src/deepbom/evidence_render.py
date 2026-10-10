"""Presentation only: receives the common engine's evidence report view on stdin."""
import io
import json
import sys
from ._report_fonts import ReportText, UNICODE_NOTE


def render(view):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, LongTable, TableStyle
    if view.get('schema') != 'deepbom.evidence_report_view.v1':
        raise ValueError('Expected the common evidence report view')
    buffer = io.BytesIO()
    styles = getSampleStyleSheet()
    text = ReportText()
    styles['BodyText'].fontName = 'DeepbomReport'
    styles['BodyText'].fontSize = 8
    styles['BodyText'].leading = 11
    def p(value):
        return Paragraph(text.safe(value if value is not None else '—'), styles['BodyText'])
    story = [Paragraph('DEEPBOM / Evidence review', styles['Title']), p(view['status']), Spacer(1, 12), p(view['validation_scope'])]
    for key in ('before', 'after', 'counts'):
        if view.get(key) is not None:
            story.extend([Spacer(1, 10), Paragraph(key.title(), styles['Heading2']), p(json.dumps(view[key], ensure_ascii=True))])
    columns = view['columns']
    data = [[p(key) for key in columns]] + [[p(row.get(key)) for key in columns] for row in view['rows']]
    table = LongTable(data, repeatRows=1, colWidths=[(landscape(A4)[0]-72)/len(columns)]*len(columns))
    table.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.whitesmoke),('VALIGN',(0,0),(-1,-1),'TOP')]))
    story.extend([Spacer(1, 14), table, Spacer(1, 14), Paragraph('Recorded source values and references', styles['Heading2'])])
    for line in json.dumps({'document':view['document'],'source_documents':view['source_documents']}, indent=2, ensure_ascii=True).splitlines():
        story.append(p(line))
    story.extend([Spacer(1, 14), p(view['boundary']), p('Result SHA-256: '+view['result_sha256'])])
    if text.escaped:
        story.append(p(UNICODE_NOTE))
    SimpleDocTemplate(buffer,pagesize=landscape(A4),leftMargin=36,rightMargin=36,topMargin=36,bottomMargin=36,title='DEEPBOM evidence review').build(story)
    return buffer.getvalue()


if __name__ == '__main__':
    try:
        raw = sys.stdin.buffer.read(16*1024*1024+1)
        if len(raw)>16*1024*1024:
            raise ValueError('Report view exceeds 16 MiB')
        sys.stdout.buffer.write(render(json.loads(raw)))
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
