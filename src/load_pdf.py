from pypdf import PdfReader
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Reads a PDF and returns a list of dicts, one per page,
    each containing the page number and extracted text.
    """
    reader = PdfReader(pdf_path)
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        pages.append({
            "source": Path(pdf_path).name,
            "page": page_num,
            "text": text if text else ""
        })

    return pages


if __name__ == "__main__":
    # Quick manual test: point this at one of your PDFs
    test_file = "data/raw/Modulkatalog-Master-Informatik.pdf"
    result = extract_text_from_pdf(test_file)

    print(f"Extracted {len(result)} pages from {test_file}\n")
    print("--- First page preview ---")
    print(result[0]["text"][:500])  # first 500 characters