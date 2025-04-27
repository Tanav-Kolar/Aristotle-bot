#importing dependencies
import pdfplumber
import re

#Load file path
file_path = "data/Nicomachean Ethics WD Ross.pdf"
full_text = ""

import pdfplumber
import re

def extract_raw_text(pdf_path: str) -> str:
    """
    Open the PDF and concatenate raw text from every page.
    """
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n".join(all_text)

def clean_text(raw_text: str) -> str:
    """
    1. Strip page headers like '12/Aristotle' or 'Nicomachean Ethics/34' at line starts.
    2. Remove all front-matter (title pages + TOC) up to the first 'BOOK I'.
    """
    #remove page numbers
    page_re = re.compile(
        r'^(?:\d+\/Aristotle|Nicomachean\s+Ethics\/\d+)\s*',
        flags=re.MULTILINE
    )
    no_headers = page_re.sub("", raw_text)

    #split into lines, find actual chapter start
    lines = no_headers.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        if re.match(r'^BOOK I\s*$', line):
            start_idx = i
            break
    #re-join from first 'BOOK I'
    return "\n".join(lines[start_idx:])

if __name__ == "__main__":
    raw = extract_raw_text("data/Nicomachean Ethics WD Ross.pdf")
    cleaned = clean_text(raw)
    with open("data/processed/nicomachean_ethics_cleaned.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)