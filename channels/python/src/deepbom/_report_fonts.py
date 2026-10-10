"""Shared monochrome PDF fonts and lossless handling of unsupported glyphs."""
import html
from pathlib import Path


class ReportText:
    def __init__(self):
        import reportlab
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        fonts = Path(reportlab.__file__).parent / 'fonts'
        names = [('DeepbomReport', 'Vera.ttf'), ('DeepbomReportBold', 'VeraBd.ttf')]
        for name, filename in names:
            if name not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(TTFont(name, str(fonts / filename)))
        self.coverage = [pdfmetrics.getFont(name).face.charWidths for name, _ in names]
        self.escaped = False

    def safe(self, value):
        output = []
        for char in str(value):
            if char in '\n\t' or all(ord(char) in coverage for coverage in self.coverage):
                output.append(char)
            else:
                self.escaped = True
                output.append(char.encode('unicode_escape').decode('ascii'))
        return html.escape(''.join(output)).replace('\n', '<br/>')


UNICODE_NOTE = ('Identifiers outside the embedded font coverage use explicit Unicode '
                'escapes. The JSON and HTML preserve their original Unicode strings.')
