# GraphRAG with Ollama

This project recreates the public [`thu-vu92/graphRAG`](https://github.com/thu-vu92/graphRAG) workflow in a cleaner folder structure and swaps the LLM layer from OpenAI to Ollama.

It keeps the original notebooks and sample outputs for reference, while adding runnable Python scripts so you can build the same style of GraphRAG pipeline with your own Ollama model.

## What is included

- `app.py`: local web app for exploring the graph, asking Ollama questions, rebuilding the graph, and scraping new data
- `start_app.bat`: one-click Windows launcher
- `src/graphrag_ollama/`: Ollama-ready Python modules for scraping, graph building, querying, and visualization
- `scripts/run_scrape.py`: collects articles and video transcripts into a CSV dataset
- `scripts/run_pipeline.py`: extracts entities and relationships, builds the graph, writes JSON/HTML, and can answer a question
- `notebooks/original/`: original notebooks copied from the source repository
- `data/sample/`: sample dataset and visualization copied from the source repository
- `visualization/graph_template.html`: the original HTML graph template

## Requirements

- Python 3.10+
- Ollama running locally or on a reachable host
- A pulled Ollama model that is good at instruction following
- A SerpAPI key for the web-scraping stage

Community detection in this version uses NetworkX Louvain clustering so it works cleanly on current Windows Python environments.

## Fastest way to run

On Windows, just double-click:

```bash
start_app.bat
```

This will:

- create `.venv` if needed
- copy `.env.example` to `.env` if needed
- install dependencies
- launch the Streamlit app

## Manual install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

## Configure

Create a `.env` file from `.env.example`.

```env
SERPAPI_KEY=your_serpapi_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_EXTRACTION_MODEL=gemma4:latest
OLLAMA_QUERY_MODEL=gemma4:latest
OLLAMA_REQUEST_TIMEOUT=300
OLLAMA_CONTEXT_WINDOW=8192
```

Note: standard local Ollama usually does not use an API key. If your Ollama server is local, `OLLAMA_BASE_URL` and model names are the important settings.

## Optional CLI commands

The app already exposes these features, but the scripts are still available:

### Run the scraper

```bash
python scripts/run_scrape.py ^
  --query "AI intellectual property" ^
  --query "copyright generative AI" ^
  --output data/ai_copyright_dataset.csv
```

### Build the GraphRAG project

```bash
python scripts/run_pipeline.py ^
  --input data/ai_copyright_dataset.csv ^
  --graph-json data/graph_data.json ^
  --graph-html data/ai_copyright_graph.html ^
  --question "What are the main legal arguments around AI copyright and training data?"
```

Outputs:

- `data/graph_data.json`
- `data/ai_copyright_graph.html`

Open the HTML file in your browser to explore the graph.

## Suggested Ollama models

The extraction stage depends on structured outputs, so instruction-tuned models work best. If one model struggles, change:

- `OLLAMA_EXTRACTION_MODEL`
- `OLLAMA_QUERY_MODEL`

Use a chat-capable Ollama model. In this workspace, `gemma4:latest` is the most reliable default.

## Source and attribution

This project is based on the original MIT-licensed repository:

- Source: [thu-vu92/graphRAG](https://github.com/thu-vu92/graphRAG)

The original notebooks, sample dataset, sample graph JSON, sample HTML output, template, and license are included here for reference.
