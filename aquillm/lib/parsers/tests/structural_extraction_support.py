"""structural extraction support: synthetic regression coverage."""

import importlib
import io
import struct
from lib.parsers.source_contracts import source_artifact_ref


def extract(name, data):
    module = importlib.import_module('lib.parsers.structural')
    return module.extract_structure(name, data, artifact=source_artifact_ref(data, member_path=None, container_sha256=None))


def workbook_bytes(rows):
    from openpyxl import Workbook
    book = Workbook()
    for row in rows:
        book.active.append(row)
    output = io.BytesIO()
    book.save(output)
    book.close()
    return output.getvalue()


def xls_bytes():
    # A minimal BIFF8 workbook; raw BIFF is supported by xlrd without an OLE wrapper.
    def record(kind, body=b''):
        return struct.pack('<HH', kind, len(body)) + body
    def bof(kind):
        return record(0x809, struct.pack('<HHHHII', 0x600, kind, 0, 0, 0, 0))
    name = b'Sheet'
    bound = lambda offset: record(0x85, struct.pack('<IBBBB', offset, 0, 0, len(name), 0) + name)
    globals_ = bof(5) + bound(0) + record(0xA)
    globals_ = bof(5) + bound(len(globals_)) + record(0xA)
    label = lambda row, col, value: record(0x204, struct.pack('<HHHHB', row, col, 0, len(value), 0) + value)
    sheet = bof(0x10) + record(0x200, struct.pack('<IIHHH', 0, 2, 0, 3, 0))
    sheet += label(0, 0, b'Product') + label(0, 1, b'Domestic') + label(0, 2, b'Foreign')
    sheet += label(1, 0, b'Widget') + record(0x203, struct.pack('<HHHd', 1, 2, 0, 7))
    sheet += record(0xE5, struct.pack('<HHHHH', 1, 0, 0, 0, 1)) + record(0xA)
    return globals_ + sheet
