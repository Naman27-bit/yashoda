---
title: yashoda-chatbot
app_file: gradio_app.py
sdk: gradio
sdk_version: 6.14.0
---
# Yashoda — Medicine Q&A (RAG) with Ollama + LangChain

A small **retrieval-augmented generation (RAG)** medical assistant that answers questions about medicines using a local dataset (`medicine_dataset.csv`).

- **Vector store**: Chroma (persistent on disk)
- **Embeddings**: Ollama `nomic-embed-text`
- **LLM**: Ollama `tinyllama`
- **UI**: Gradio (optional)

> ⚠️ Note: This tool is for information only. It is **not medical advice**.

---

## Project Structure

- `vector.py`  
  Loads/creates a Chroma DB from `medicine_dataset.csv` and exposes `retriever`.

- `main.py`  
  Command-line chat loop using the RAG chain.

- `gradio_app.py`  
  Gradio web interface for the same RAG flow.

- `medicine_dataset.csv`  
  Source data used to build the knowledge base.

- `chrome_langchain_db/`  
  Persisted Chroma vector database.

---

## Prerequisites

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Install and run Ollama

You must have **Ollama** installed and running locally.

Then pull the models used by this project:

```bash
ollama pull tinyllama
ollama pull nomic-embed-text
```

---

## Data Requirements

`vector.py` reads `medicine_dataset.csv` and expects these columns:

- `id`
- `name`
- `substitute0`
- `substitute1`
- `substitute2`

If your CSV uses different column names, update `vector.py` accordingly.

---

## How it works (high level)

1. **Build/Load vector DB** (`vector.py`)
   - CSV rows are embedded with `nomic-embed-text`
   - Stored in a persistent Chroma collection (`medicine_collection`)

2. **Retrieve** (`retriever.invoke(question)`)
   - Finds the most relevant entries for the user question

3. **Generate**
   - `tinyllama` is prompted with retrieved context
   - Produces an answer grounded in that context

---

## Run the project

### Option A — Command Line

```bash
python main.py
```

You’ll be prompted in the terminal. Type `q` to quit.

### Option B — Gradio Web UI

```bash
python gradio_app.py
```

Then open the URL shown in the terminal.

---

## Files you may need to edit

- `vector.py`
  - To change the dataset path, Chroma location, or embedding settings.
- `main.py` / `gradio_app.py`
  - To adjust prompt template or model name.

---

## Known Limitations

- Answers depend on the quality and coverage of `medicine_dataset.csv`.
- The prompt currently uses a general template and may not always follow medical best practices.
- For safety-critical use, consult a qualified healthcare professional.