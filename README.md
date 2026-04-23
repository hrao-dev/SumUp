# SumUp — Document Intelligence Platform

> Distill any document, article, or text into a structured summary — with keyword extraction, named entity recognition, ROUGE scoring, sentence highlighting, and reading level analysis.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HF%20Space-harao--ml%2FSumUp-blue)](https://huggingface.co/spaces/harao-ml/SumUp)
[![Model](https://img.shields.io/badge/🤗%20Model-flant5--finetuned--summarize-green)](https://huggingface.co/harao-ml/flant5-finetuned-summarize)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)](https://python.org)

---

## Overview

**SumUp** is a full-stack NLP web application that summarizes any long-form content — pasted text, news article URLs, or uploaded PDF/TXT files — using a fine-tuned FLAN-T5 model. Beyond summarization, it runs a full downstream analysis pipeline covering keyword extraction, named entity recognition, ROUGE evaluation, extractive sentence highlighting, and readability scoring.

The app is built with **Streamlit** and deployed as a **Dockerized Hugging Face Space**.

---

## Features

| Feature | Description |
|---|---|
| **AI Summarization** | Fine-tuned FLAN-T5 (`harao-ml/flant5-finetuned-summarize`) generates abstractive summaries at three configurable lengths |
| **URL Article Fetching** | Paste any news or article URL — `newspaper4k` fetches and parses the full text, author, and publish date automatically |
| **PDF & TXT Upload** | Upload documents directly; PyPDF2 extracts full text from multi-page PDFs |
| **Keyword Extraction** | KeyBERT extracts the top 8 key phrases (1–2 gram) using semantic similarity |
| **Named Entity Recognition** | spaCy `en_core_web_sm` identifies People, Organizations, Locations, Dates, Money, Events, Products, and Laws |
| **ROUGE Scoring** | ROUGE-1, ROUGE-2, and ROUGE-L F1/Precision/Recall scores quantify how much source content is preserved |
| **Sentence Highlighting** | The top 3 source sentences most aligned with the summary are highlighted inline, scored by ROUGE-1 |
| **Reading Level Analysis** | Flesch-Kincaid grade and reading ease scores compared between source and summary, plus compression ratio |
| **Configurable Length** | Choose between Brief (30–80 tokens), Balanced (60–180 tokens), or Detailed (120–300 tokens) summaries |
| **Dark UI** | Custom-styled Streamlit interface with DM Sans/DM Mono typography and a dark design system |

---

## Tech Stack

| Layer | Library / Tool |
|---|---|
| UI Framework | Streamlit |
| Summarization Model | `transformers` 4.44.0 · `harao-ml/flant5-finetuned-summarize` (FLAN-T5) |
| Deep Learning | PyTorch 2.3.1 |
| Keyword Extraction | KeyBERT 0.8.5 · sentence-transformers 2.7.0 |
| NER | spaCy 3.7.5 · `en_core_web_sm` |
| Evaluation | rouge-score |
| Readability | textstat |
| Article Scraping | newspaper4k · lxml-html-clean |
| PDF Parsing | PyPDF2 |
| Deployment | Docker · Hugging Face Spaces |

---

## Project Structure

```
SumUp/
├── app.py              # Main Streamlit application (UI + NLP pipeline)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container build instructions for HF Spaces
├── README.md           # This file
└── src/                # (Reserved for future modular source files)
```

---

## How It Works

### Input
The sidebar accepts three mutually exclusive input types:
1. **Pasted text** — any raw long-form content
2. **URL** — detected automatically by `http://` / `https://` prefix; article body, title, author, and date are fetched with `newspaper4k`
3. **File upload** — `.pdf` or `.txt` files; PDF text is extracted page-by-page with PyPDF2

### Processing Pipeline
Once **Summarize** is clicked, the app runs five sequential steps:

```
Input text
    │
    ▼
clean_text()          → collapse whitespace, remove noise
split_text()          → chunk into 512-token windows
    │
    ▼
generate_summary()    → FLAN-T5 abstractive summarization per chunk
    │
    ├──► extract_insights()   → KeyBERT (keywords) + spaCy (NER)
    ├──► compute_rouge()      → ROUGE-1 / ROUGE-2 / ROUGE-L
    ├──► highlight_sentences()→ top-3 extractive sentences by ROUGE-1
    └──► reading_level()      → Flesch-Kincaid grade + ease + compression %
```

### Output Panels
Results are displayed in four sections in the main area:

1. **Summary** — the generated abstractive summary (with article metadata if URL-sourced)
2. **Information Extraction** — key phrases (tag pills) and named entities grouped by type
3. **ROUGE Evaluation** — F1, Precision, Recall for R-1, R-2, R-L with ASCII bar visualizations
4. **Reading Level & Compression** — four metric cards (Source Grade, Summary Grade, Reading Ease, Compression %)

---

## Running Locally

### Prerequisites
- Python 3.10+
- pip

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/SumUp.git
cd SumUp

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will be available at `http://localhost:8501`.

> **Note:** On first run, the FLAN-T5 model (~300 MB) and spaCy model will be downloaded automatically and cached. This may take a few minutes.

---

## Running with Docker

```bash
# Build the image
docker build -t sumup .

# Run the container
docker run -p 7860:7860 sumup
```

The app will be available at `http://localhost:7860`.

---

## Model

The summarization backbone is [`harao-ml/flant5-finetuned-summarize`](https://huggingface.co/harao-ml/flant5-finetuned-summarize), a FLAN-T5 model fine-tuned for abstractive summarization tasks. It is loaded via the Hugging Face `transformers` pipeline and cached with `@st.cache_resource` to avoid reloading across sessions.

Long documents are handled by splitting the input into 512-token chunks and concatenating the per-chunk summaries.

---

## Configuration

Summary length is controlled at inference time via `min_length` / `max_length` parameters:

| Mode | min_length | max_length |
|---|---|---|
| Brief | 30 | 80 |
| Balanced | 60 | 180 |
| Detailed | 120 | 300 |

---

## Live Demo

Try it on Hugging Face Spaces: [https://huggingface.co/spaces/harao-ml/SumUp](https://huggingface.co/spaces/harao-ml/SumUp)

---

## License

This project is open source. See [LICENSE](LICENSE) for details.

---
