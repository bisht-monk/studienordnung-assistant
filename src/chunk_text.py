from src.load_pdf import extract_text_from_pdf


def chunk_pages(pages: list[dict], chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    """
    Takes the list of page dicts from load_pdf and splits each page's text
    into smaller overlapping chunks, so each chunk is small enough for an LLM
    to use effectively, while keeping some shared context between chunks.
    """
    chunks = []

    for page in pages:
        text = page["text"]

        if not text.strip():
            continue  # skip empty pages

        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "text": chunk_text.strip()
            })

            start += chunk_size - overlap  # move forward, but overlap a bit

    return chunks


import json

if __name__ == "__main__":
    pdf_path = "data/raw/Modulkatalog-Master-Informatik.pdf"
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_pages(pages)

    print(f"Created {len(chunks)} chunks from {len(pages)} pages")

    output_path = "data/processed/chunks.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved chunks to {output_path}")