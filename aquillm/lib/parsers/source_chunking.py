"""Token-budgeted canonical slices with separate embedding context.

Callers supply a verified local tokenizer adapter exposing a nonempty immutable
``revision`` and either ``encode(text)`` or ``__call__(text)``. No tokenizer is
downloaded or selected here. Existing configured chunk defaults stay unchanged.
"""
from hashlib import sha256

from lib.embeddings.provenance import input_digest
from .canonical_projection import PROJECTION_POLICY, canonical_json, validate_extraction, visible
from .source_contracts import SourceChunkV1, SourceContractError


# These bound only the extra exhaustive search required when checked heuristic
# probes find no nonempty fit. They are offline work limits, not token budgets.
MAX_FALLBACK_TOKENIZER_CALLS = 4096
MAX_FALLBACK_INPUT_CODEPOINTS = 1_000_000
FITTING_POLICY = 'checked-heuristic-bounded-nonmonotone-fallback-v1'


def _validate_options(tokenizer, max_tokens, overlap_tokens):
    if type(max_tokens) is not int or max_tokens <= 0 or type(overlap_tokens) is not int or not 0 <= overlap_tokens < max_tokens:
        raise SourceContractError('invalid_chunk_token_budget')
    revision = getattr(tokenizer, 'revision', None)
    if type(revision) is not str or not revision or not (callable(tokenizer) or callable(getattr(tokenizer, 'encode', None))):
        raise SourceContractError('verified_local_tokenizer_required')
    return revision


def chunker_key_for(*, tokenizer, max_tokens: int, overlap_tokens: int) -> str:
    revision = _validate_options(tokenizer, max_tokens, overlap_tokens)
    payload = {'version': 'canonical-source-chunker-v1', 'max_tokens': max_tokens,
               'overlap_tokens': overlap_tokens, 'tokenizer_revision': revision,
               'projection_policy': PROJECTION_POLICY, 'context_policy': 'heading-column-formula-schema-v1',
               'fitting_policy': FITTING_POLICY, 'max_fallback_tokenizer_calls': MAX_FALLBACK_TOKENIZER_CALLS,
               'max_fallback_input_codepoints': MAX_FALLBACK_INPUT_CODEPOINTS}
    return 'canonical-source-v1:' + sha256(canonical_json(payload)).hexdigest()


def chunk_extraction(extraction, *, chunker_key: str, tokenizer, max_tokens: int, overlap_tokens: int) -> tuple[SourceChunkV1, ...]:
    validate_extraction(extraction)
    if chunker_key != chunker_key_for(tokenizer=tokenizer, max_tokens=max_tokens, overlap_tokens=overlap_tokens):
        raise SourceContractError('chunker_key_mismatch')
    encode = getattr(tokenizer, 'encode', None) or tokenizer
    def count(text):
        try:
            return len(encode(text))
        except Exception as exc:
            raise SourceContractError('local_tokenizer_failed') from exc
    fallback_calls, fallback_codepoints = 0, 0
    def fallback_count(text):
        nonlocal fallback_calls, fallback_codepoints
        if (fallback_calls + 1 > MAX_FALLBACK_TOKENIZER_CALLS
                or fallback_codepoints + len(text) > MAX_FALLBACK_INPUT_CODEPOINTS):
            # A stopped search proves no absence. Nothing is published until
            # this function's final tuple return, including on capacity failure.
            raise SourceContractError('chunker_work_limit')
        fallback_calls += 1
        fallback_codepoints += len(text)
        return count(text)
    chunks, heading = [], ''
    for element in extraction.elements:
        if element.kind == 'figure':
            continue
        source = extraction.canonical_text
        context = ''
        if element.kind == 'heading':
            heading = source[element.start:element.end].rstrip('\n')
        elif heading:
            context += f'heading={heading}\n'
        if element.cells:
            columns = ' '.join(f'{cell.column}:{visible("|".join(cell.header_keys))}' for cell in element.cells if cell.header_keys)
            if columns:
                context += f'columns={columns}\n'
            formulas = ' '.join(f'{cell.address} formula={visible(cell.formula)} value={"<unknown>" if cell.value_status == "uncached" else visible(cell.text)}'
                                for cell in element.cells if cell.formula is not None)
            if formulas:
                context += f'formula_context={formulas}\n'
        start = element.start
        while start < element.end:
            # Binary probes are only a fast candidate heuristic: BPE token counts
            # are neither additive nor monotone over Unicode slice boundaries.
            # Every accepted candidate uses the actual complete embedding input.
            end = element.end
            if count(context + source[start:end]) > max_tokens:
                low, high = start + 1, element.end
                best = None
                checked = {element.end}
                while low <= high:
                    middle = (low + high) // 2
                    checked.add(middle)
                    if count(context + source[start:middle]) <= max_tokens:
                        best, low = middle, middle + 1
                    else:
                        high = middle - 1
                if best is None:
                    for candidate in range(element.end - 1, start, -1):
                        if candidate not in checked and fallback_count(context + source[start:candidate]) <= max_tokens:
                            best = candidate
                            break
                    if best is None:
                        raise SourceContractError('unrepresentable_source_element')
                end = best
            text = source[start:end]
            embedding = context + text
            if count(embedding) > max_tokens:
                raise SourceContractError('unrepresentable_source_element')
            chunks.append(SourceChunkV1(start, end, text, embedding, (element.element_key,), sha256(text.encode('utf-8')).hexdigest(), input_digest(embedding)))
            if end == element.end:
                break
            next_start = end
            # Retain a source suffix bounded by the requested overlap count.
            # The following chunk still counts its complete embedding context.
            if overlap_tokens:
                low, high = start + 1, end
                checked = set()
                while low <= high:
                    middle = (low + high) // 2
                    checked.add(middle)
                    if count(source[middle:end]) <= overlap_tokens:
                        next_start, high = middle, middle - 1
                    else:
                        low = middle + 1
                if next_start == end:
                    for candidate in range(start + 1, end):
                        if candidate not in checked and fallback_count(source[candidate:end]) <= overlap_tokens:
                            next_start = candidate
                            break
            if next_start <= start:
                raise SourceContractError('unrepresentable_source_element')
            start = next_start
    return tuple(chunks)
