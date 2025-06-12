import re
import json

# Load cleaned text data
with open("data/processed/nicomachean_ethics_cleaned.txt", "r", encoding="utf-8") as f:
    text = f.read()


def parse_corpus(text: str) -> list[dict]:
    """
    Splits the cleaned Nicomachean Ethics text into books and sections.
    Returns a list of dicts with keys: book, section, text, char_length, tokens.
    """
    entries = []

    # locate all 'BOOK <Roman>' markers
    book_pattern = r"\bBOOK\s+([IVXL]+)\b"
    book_matches = list(re.finditer(book_pattern, text))

    for idx, bm in enumerate(book_matches):
        book_id = bm.group(1)
        start = bm.end()
        end = book_matches[idx + 1].start() if idx + 1 < len(book_matches) else len(text)
        book_text = text[start:end].strip()

        #split by lines that start with a section number, e.g. "\n1\n"
        sections = re.split(r"\n(?=\d+\n)", book_text)
        for sec in sections:
            m = re.match(r"(\d+)\n(.*)", sec, re.DOTALL)
            if not m:
                continue
            sec_num  = m.group(1)
            sec_text = m.group(2).strip()

            entries.append({
                "book":        book_id,
                "section":     sec_num,
                "text":        sec_text,
                "char_length": len(sec_text),
                "tokens":      len(sec_text.split())  # simple whitespace token count
            })

    return entries

jsonified = parse_corpus(text)

with open("data/processed/nicomachean_ethics_sections.json", "w", encoding="utf-8") as f:
    json.dump(jsonified, f, indent=2)
