# Sample Documents

This directory is intended for sample documents used to test and demonstrate the SecureAgentRAG ingestion pipeline.

## Supported Formats

- **PDF** (`.pdf`) — Parsed with PyMuPDF; OCR fallback via PaddleOCR for scanned pages
- **DOCX** (`.docx`) — Parsed with python-docx
- **Images** (`.png`, `.jpg`, `.jpeg`) — OCR via PaddleOCR

## What to Put Here

Add sample documents that demonstrate:

1. **English text documents** — For testing standard retrieval
2. **Arabic text documents** — For testing bilingual support
3. **Mixed-language documents** — For testing language detection
4. **Scanned PDFs / images** — For testing OCR pipeline
5. **Multi-page documents** — For testing chunking strategies
6. **Documents with different access levels** — For testing RBAC metadata tagging

## Notes

- Do **not** commit sensitive or proprietary documents
- Keep sample files small (< 5MB each) for fast CI/CD
- Actual sample files will be added in subsequent development tasks
