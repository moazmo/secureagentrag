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

## Bundled Samples

| File | Purpose |
|------|---------|
| `sample_english.txt` | Corporate AI governance / RBAC policy (English) |
| `sample_arabic.txt` | Privacy policy excerpt (Arabic) |
| `sample_mixed.txt` | Bilingual document for language-detection chunking |
| `sample_internal_report.pdf` | Multi-line PDF for the PyMuPDF text-extraction path |
| `sample_invoice.png` | Synthesized invoice image for the PaddleOCR fallback path |
| `demo_rbac/*.txt` | 9 English RBAC demo docs (handbook, runbook, finance, security policy, vendor MSA, …) ingested into the live `documents` collection with per-doc sensitivity + roles. |
| `arabic_eg/*.txt` | 8 illustrative Arabic Egypt civic docs (rental contract, labor law, VAT, HR policy, tenant rights, consumer protection, freelance tax, social insurance) — the "افهم عقدك" flagship corpus. LOW + broad roles (HR policy MEDIUM). |
| `real/NIST_AI_RMF.pdf` | The real NIST AI RMF for in-domain retrieval + rerank benchmarking. |

## Notes

- Do **not** commit sensitive or proprietary documents
- Keep sample files small (< 5MB each) for fast CI/CD
- These files are illustrative test fixtures, not real organizational data
