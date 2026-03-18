from json import dumps, loads
from pypdf import PdfReader
from pprint import pp
from pydantic import BaseModel, TypeAdapter
from typing import Optional
from os import getenv
from pdf2image import convert_from_path
from functools import reduce
import re
import anthropic
import base64
import io


class Chunk(BaseModel):
    text: str
    footnotes: list[str]

class ChunkingResponse(BaseModel):
    expanded_chunk: Optional[Chunk]
    new_chunks: list[Chunk]


client = anthropic.Anthropic()

def do_page_with_llm(page, last_chunk: Optional[Chunk]) -> ChunkingResponse:

    def clean_line_breaks(text: str) -> str:
        patterns = [(r'-\n(?!\n)', ''),
                    (r'(?<!\n)\n(?!\n)', ' ')]
        ret = reduce(lambda t, pattern: re.sub(pattern[0], pattern[1], t), patterns, text)
        return ret
    system = """
You are tasked with chunking a PDF page's body text for database ingestion. Follow these guidelines:

1. **Chunking:**
   - Divide the body text into consecutive chunks of 250–1000 words each.
   - Ensure every word is included (no gaps between chunks).
   - Break chunks at natural semantic boundaries (e.g., between paragraphs or sections).

2. **Handling Continuations:**
   - If the current page begins with text that continues from the provided `last_chunk`, merge this text with `last_chunk` into an `expanded_chunk`.
   - Do not include the merged text again in `new_chunks`.

3. **Text Integrity:**
   - Use only the verbatim text (no paraphrasing or editorial additions).
   - Exclude any author lists and bibliography sections.
   - Do not add titles or word counts to any chunk.

4. **Footnotes:**
   - Include the full text of any footnotes in the chunk where they are referenced (use the footnote text, not the number).
   - Do not include citations as footnotes.

5. **Formatting:**
   - Use line breaks to represent paragraph breaks. REMOVE LINE BREAKS WHERE THEY RESULT FROM TEXT WRAPPING WITHIN A BLOCK OF TEXT.
   - Always represent paragraph breaks with double line breaks.
   - Where words are hyphenated at line wraps, join the words without a hyphen.
   - Use markdown to represent headings, using only the `#` character for headings of any level.
   - Use markdown to represent italics, boldface, lists, and bullet points.
   - Use HTML tags for tables.

Return the output as a JSON object with the following structure:

{
    "expanded_chunk": {
        "text": "Text combining last_chunk with the start of the current page",
        "footnotes": ["Text of footnote 1", "Text of footnote 2", ...]
    } or null,
    "new_chunks": [
        {
            "text": "First complete chunk of text",
            "footnotes": ["Text of footnote 3", "Text of footnote 4", ...]
        },
        {
            "text": "Second chunk of text",
            "footnotes": []
        },
        ...
    ]
}

    """
    # Convert PIL image to base64
    buf = io.BytesIO()
    page.save(buf, format='PNG')
    image_data = base64.b64encode(buf.getvalue()).decode('utf-8')

    content = []
    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}})
    if last_chunk:
        content.append({"type": "text", "text": last_chunk.model_dump_json()})

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": content}],
    )

    parsed = ChunkingResponse.model_validate_json(response.content[0].text)
    parsed.new_chunks = list([Chunk(text=clean_line_breaks(chunk.text), footnotes=chunk.footnotes) for chunk in parsed.new_chunks])
    if parsed.expanded_chunk:
        parsed.expanded_chunk = Chunk(text=clean_line_breaks(parsed.expanded_chunk.text), footnotes=parsed.expanded_chunk.footnotes)
    return parsed


images = convert_from_path("/home/chandler/software1-3.pdf")

ret: list[Chunk] = []
for page in images:
    print("doing page")
    response = do_page_with_llm(page, ret[-1] if ret else None)
    if response.expanded_chunk and ret:
        ret[-1] = response.expanded_chunk
    ret += response.new_chunks

with open("/home/chandler/chunked.json", "w") as f:
    f.write(dumps([chunk.model_dump() for chunk in ret]))
