"""Score real canonical facts and windows; embedding context is not source proof."""


class SyntheticByteTokenizer:
    """Locally exact UTF-8 byte tokens; no model-quality/tokenizer claim."""

    revision = 'synthetic-utf8-byte-tokens-v1'

    @staticmethod
    def encode(text):
        return text.encode('utf-8')


def tally(matched, expected):
    return {'matched': matched, 'expected': expected,
            'recall': matched / expected if expected else None}


def fact_spans(extraction, fact):
    text, needle = extraction.canonical_text, fact['text']
    if not needle:
        raise ValueError('empty_fixture_fact')
    position = 0
    spans = []
    while (position := text.find(needle, position)) >= 0:
        end = position + len(needle)
        for element in extraction.elements:
            if (element.start <= position and end <= element.end
                    and all(getattr(element, key) == value
                            for key, value in fact['region'].items())):
                spans.append((position, end, element))
        position += 1
    return spans


def measure_extraction(extraction, expected):
    facts = tally(sum(bool(fact_spans(extraction, f)) for f in expected['facts']),
                  len(expected['facts']))
    matched_cells = 0
    for wanted in expected['cells']:
        matched_cells += any(
            all((getattr(element, key) if key in ('sheet', 'page', 'slide')
                 else list(cell.header_keys) if key == 'header_keys'
                 else getattr(cell, key)) == value for key, value in wanted.items())
            for element in extraction.elements for cell in element.cells
        )
    cells = tally(matched_cells, len(expected['cells']))
    quality = (extraction.quality == expected['quality']
               and list(extraction.issue_codes) == expected['issues'])
    return {'facts': facts, 'cells': cells, 'quality_and_issues_preserved': quality,
            'passed': quality and facts['matched'] == facts['expected']
            and cells['matched'] == cells['expected']}


def _union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def measure_windows(extraction, windows, facts):
    text = extraction.canonical_text
    valid = [w for w in windows if type(w.start) is int and type(w.end) is int
             and 0 <= w.start < w.end <= len(text)
             and w.source_text == text[w.start:w.end]]
    intervals = _union((w.start, w.end) for w in valid)
    coverage = sum(end - start for start, end in intervals)
    encode = SyntheticByteTokenizer.encode
    source_tokens = sum(len(encode(w.source_text)) for w in valid)
    unique_tokens = sum(len(encode(text[start:end])) for start, end in intervals)
    hits, precisions = 0, []
    for fact in facts:
        candidates = []
        for start, end, element in fact_spans(extraction, fact):
            for window in valid:
                if window.start <= start and end <= window.end:
                    overlap = min(window.end, element.end) - max(window.start, element.start)
                    candidates.append(overlap / (window.end - window.start))
        if candidates:
            hits += 1
            precisions.append(max(candidates))
        else:
            precisions.append(0.0)
    return {
        'window_count': len(windows), 'invalid_windows': len(windows) - len(valid),
        'source_coverage': coverage / len(text) if text else None,
        'support_spans': tally(hits, len(facts)),
        'source_region_precision': sum(precisions) / len(facts) if facts else None,
        'source_region_recall': hits / len(facts) if facts else None,
        'source_tokens': source_tokens,
        'redundant_source_tokens': source_tokens - unique_tokens,
        'embedding_tokens': sum(len(encode(w.embedding_text)) for w in valid),
        'embedding_context_tokens': sum(len(encode(w.embedding_text)) for w in valid) - source_tokens,
    }
