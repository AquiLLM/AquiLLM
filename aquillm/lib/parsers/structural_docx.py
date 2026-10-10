"""Ordered native DOCX traversal, including nested tables and controls."""

import io


def extract_docx(builder, data, *, _cell, _headers):
    from docx import Document
    from docx.table import Table, _Cell
    from docx.text.paragraph import Paragraph
    from docx.oxml.ns import qn
    doc = Document(io.BytesIO(data))
    def unavailable(node):
        if (node.tag == qn('w:altChunk') or any(next(node.iter(qn(tag)), None) is not None
                for tag in ('w:t', 'w:delText', 'w:instrText', 'a:t', 'a:blip'))):
            builder.issue('document_wrapper_structure_unavailable')

    # python-docx omits these wrappers from Paragraph.text/Table.rows. Native
    # body/cell content controls are supported below; run/row/cell controls and
    # revision/custom XML wrappers must not silently advertise completeness.
    block_parents = {qn('w:body'), qn('w:tc'), qn('w:sdtContent')}
    unsupported_wrappers = {qn(f'w:{name}') for name in
                            ('customXml', 'ins', 'del', 'moveFrom', 'moveTo', 'altChunk')}
    for node in doc.element.body.iter():
        if node.tag in unsupported_wrappers or (node.tag == qn('w:sdt') and node.getparent().tag not in block_parents):
            unavailable(node)

    def blocks(owner):
        # Unwrap controls in place, without descendant queries that conflate
        # table/cell order. An explicit stack also handles deeply nested controls.
        stack = [iter(owner)]
        while stack:
            child = next(stack[-1], None)
            if child is None:
                stack.pop()
            elif child.tag in (qn('w:p'), qn('w:tbl')):
                yield child
            elif child.tag == qn('w:sdt'):
                contents = child.findall(qn('w:sdtContent'))
                if len(contents) != 1:
                    builder.issue('document_wrapper_structure_unavailable')
                else:
                    stack.append(iter(contents[0]))
            else:
                unavailable(child)

    for relation in doc.part.rels.values():
        if relation.reltype.rsplit('/', 1)[-1] in {'header', 'footer', 'footnotes', 'endnotes', 'comments'}:
            element = getattr(relation.target_part, 'element', None)
            if element is None or any(node.text for node in element.iter(qn('w:t'))):
                builder.issue('auxiliary_document_content_unavailable')
    def paragraph(node, owner, parent=None):
        p = Paragraph(node, owner)
        for child in node:
            if child.tag not in (qn('w:r'), qn('w:hyperlink'), qn('w:pPr')):
                unavailable(child)
        key = parent
        if p.text:
            kind = 'heading' if p.style and p.style.name.startswith('Heading') else 'paragraph'
            key = builder.add(p.text, kind, parent_key=parent)
        for blip in node.iter(qn('a:blip')):
            builder.figure(parent_key=key)
            builder.issue('figure_layout_unavailable')
        return key
    def table(node, owner, parent=None):
        table_obj = Table(node, owner)
        for child in node:
            if child.tag not in (qn('w:tr'), qn('w:tblPr'), qn('w:tblGrid')):
                unavailable(child)
        headers = ()
        for row_number, row in enumerate(table_obj.rows, 1):
            for child in row._tr:
                if child.tag not in (qn('w:tc'), qn('w:trPr')):
                    unavailable(child)
            cells, tails, column = [], [], 1
            # Read original tc nodes instead of row.cells, which aliases merged
            # cells and repeats the merged origin's text in covered coordinates.
            before = row._tr.grid_before
            for col in range(1, before + 1):
                cells.append(_cell(row_number, col))
            column += before
            for tc in row._tr.tc_lst:
                cell = _Cell(tc, table_obj)
                span = tc.grid_span
                leading, tail, seen_nested = [], [], False
                for child in blocks(tc):
                    if child.tag == qn('w:tbl'):
                        seen_nested = True
                    if child.tag in (qn('w:p'), qn('w:tbl')):
                        if seen_nested:
                            tail.append(child)
                        else:
                            leading.append(Paragraph(child, cell).text)
                value = '\n'.join(leading)
                # A vertical continuation is an explicit blank coordinate.
                if tc.vMerge == 'continue':
                    value = None
                row_span = 1
                if tc.vMerge == 'restart':
                    for following in table_obj.rows[row_number:]:
                        next_column = following._tr.grid_before + 1
                        continuation = None
                        for following_tc in following._tr.tc_lst:
                            if next_column == column:
                                continuation = following_tc
                                break
                            next_column += following_tc.grid_span
                        if continuation is None or continuation.vMerge != 'continue' or continuation.grid_span != span:
                            break
                        row_span += 1
                cells.append(_cell(row_number, column, value, row_span=row_span, column_span=span,
                                   headers=headers[column - 1] if row_number > 1 and column <= len(headers) else ()))
                for covered in range(1, span):
                    cells.append(_cell(row_number, column + covered))
                tails.append((cell, tail, column, span))
                column += span
            for _ in range(row._tr.grid_after):
                cells.append(_cell(row_number, column))
                column += 1
            if row_number == 1:
                headers = _headers([cell.text for cell in cells])
            if not any(tail for _, tail, _, _ in tails):
                key = builder.row(tuple(cells), parent_key=parent)
                for cell, _, _, _ in tails:
                    for child in blocks(cell._tc):
                        if child.tag == qn('w:p'):
                            for blip in child.iter(qn('a:blip')):
                                builder.figure(parent_key=key)
                                builder.issue('figure_layout_unavailable')
            else:
                # Nested blocks belong at their exact position within each cell.
                # Emit per-cell row segments only where nesting requires it;
                # an ordinary row remains one full coordinate-bearing element.
                emitted_columns = set()
                for cell, tail, cell_column, span in tails:
                    segment = tuple(c for c in cells if cell_column <= c.column < cell_column + span)
                    emitted_columns.update(c.column for c in segment)
                    key = builder.row(segment, parent_key=parent)
                    for child in blocks(cell._tc):
                        if child in tail:
                            break
                        if child.tag == qn('w:p'):
                            for blip in child.iter(qn('a:blip')):
                                builder.figure(parent_key=key)
                                builder.issue('figure_layout_unavailable')
                    for child in tail:
                        if child.tag == qn('w:tbl'):
                            table(child, cell, key)
                        else:
                            paragraph(child, cell, key)
                omitted = tuple(c for c in cells if c.column not in emitted_columns)
                if omitted:
                    builder.row(omitted, parent_key=parent)
    for node in blocks(doc.element.body):
        if node.tag == qn('w:p'):
            paragraph(node, doc)
        elif node.tag == qn('w:tbl'):
            table(node, doc)
