from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1] / "backend"
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from graphrag_ollama.app_support import (
    answer_question_from_graph,
    ensure_env_file,
    get_active_dataset_path,
    get_active_graph_paths,
    graph_stats,
    load_graph_data,
)
from graphrag_ollama.config import DEFAULT_GRAPH_HTML_PATH, DEFAULT_GRAPH_JSON_PATH, load_runtime_config
from graphrag_ollama.graph_pipeline import run_pipeline
from graphrag_ollama.scrape import run_scrape


st.set_page_config(
    page_title="GraphRAG Studio",
    page_icon="graph",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
        :root {
            --ink: #11211d;
            --teal: #0d8f7a;
            --sand: #f5efe2;
            --card: rgba(255,255,255,0.72);
            --line: rgba(17,33,29,0.12);
            --accent: #ff8a3d;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(13,143,122,0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(255,138,61,0.18), transparent 24%),
                linear-gradient(180deg, #fbf7ee 0%, #eef7f4 100%);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }
        [data-testid="stSidebar"] {
            background: rgba(255,255,255,0.82);
            border-right: 1px solid var(--line);
        }
        .hero {
            padding: 1.4rem 1.6rem;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(13,143,122,0.94), rgba(17,33,29,0.94));
            color: #f8faf8;
            box-shadow: 0 24px 60px rgba(17,33,29,0.16);
            margin-bottom: 1rem;
        }
        .hero h1 {
            font-size: 2.5rem;
            margin: 0;
            letter-spacing: -0.04em;
        }
        .hero p {
            margin: 0.55rem 0 0;
            font-size: 1rem;
            max-width: 52rem;
        }
        .metric-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1rem 1.1rem;
            backdrop-filter: blur(14px);
            min-height: 108px;
        }
        .metric-label {
            color: rgba(17,33,29,0.72);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }
        .soft-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1.15rem;
            backdrop-filter: blur(14px);
        }
        .hint {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.84rem;
            color: rgba(17,33,29,0.75);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset_preview(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def main() -> None:
    apply_theme()
    env_path = ensure_env_file()
    runtime = load_runtime_config()
    graph_json_path, graph_html_path = get_active_graph_paths()
    dataset_path = get_active_dataset_path()
    graph_data = load_graph_data(graph_json_path)
    stats = graph_stats(graph_data)

    st.markdown(
        """
        <section class="hero">
            <h1>GraphRAG Studio</h1>
            <p>
                Explore the AI copyright knowledge graph, ask Ollama questions about it,
                and rebuild the graph from your dataset without touching raw notebooks.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    left, mid, right, extra = st.columns(4)
    with left:
        render_metric("Nodes", str(stats["nodes"]))
    with mid:
        render_metric("Links", str(stats["links"]))
    with right:
        render_metric("Communities", str(stats["communities"]))
    with extra:
        render_metric("Ollama Model", runtime.query_model)

    st.sidebar.title("Control Room")
    st.sidebar.markdown(f"`.env` file: `{env_path}`")
    st.sidebar.markdown(f"Graph JSON: `{graph_json_path.name}`")
    st.sidebar.markdown(f"Graph HTML: `{graph_html_path.name}`")
    st.sidebar.markdown(f"Dataset: `{dataset_path.name}`")
    st.sidebar.markdown(f"Ollama base URL: `{runtime.ollama_base_url}`")
    st.sidebar.markdown(
        "SerpAPI key status: "
        + ("configured" if runtime.serpapi_key else "missing, so scrape is optional right now")
    )

    explore_tab, ask_tab, build_tab, scrape_tab = st.tabs(
        ["Explore Graph", "Ask Ollama", "Build Graph", "Scrape Data"]
    )

    with explore_tab:
        st.markdown("### Interactive Graph")
        st.markdown(
            '<div class="soft-card"><span class="hint">The app opens the generated graph if present, otherwise it falls back to the included sample graph.</span></div>',
            unsafe_allow_html=True,
        )
        html = graph_html_path.read_text(encoding="utf-8")
        components.html(html, height=760, scrolling=True)

        preview_df = load_dataset_preview(dataset_path)
        st.markdown("### Dataset Preview")
        st.dataframe(preview_df.head(25), use_container_width=True)

    with ask_tab:
        st.markdown("### Ask Questions About the Graph")
        st.markdown(
            '<div class="soft-card"><span class="hint">This uses your local Ollama model and the active graph JSON. It works immediately with the bundled sample graph.</span></div>',
            unsafe_allow_html=True,
        )
        default_question = "What are the main legal arguments around AI copyright and training data?"
        question = st.text_area("Question", value=default_question, height=120)
        if st.button("Ask Ollama", type="primary", use_container_width=True):
            with st.spinner("Ollama is reading the graph and drafting an answer..."):
                answer = answer_question_from_graph(question, graph_data)
            st.markdown("### Answer")
            st.write(answer)

    with build_tab:
        st.markdown("### Build or Rebuild the Graph")
        dataset_input = st.text_input("Dataset CSV", value=str(dataset_path))
        max_articles = st.slider("How many articles to use", min_value=1, max_value=50, value=3)
        build_question = st.text_input(
            "Optional question after build",
            value="Which organizations and legal cases stand out most in this graph?",
        )
        if st.button("Build Graph with Ollama", use_container_width=True):
            with st.spinner("Building the graph. This can take a few minutes on local models..."):
                run_pipeline(
                    input_csv=Path(dataset_input),
                    graph_json_path=DEFAULT_GRAPH_JSON_PATH,
                    graph_html_path=DEFAULT_GRAPH_HTML_PATH,
                    question=build_question,
                    max_articles=max_articles,
                )
            st.success("Graph build finished. Reloading the latest graph files now.")
            st.rerun()

    with scrape_tab:
        st.markdown("### Scrape New Source Material")
        st.markdown(
            '<div class="soft-card"><span class="hint">This step needs a SerpAPI key in your `.env`. If you skip it, the app still works with the bundled sample dataset.</span></div>',
            unsafe_allow_html=True,
        )
        query_1 = st.text_input("Query 1", value="AI intellectual property")
        query_2 = st.text_input("Query 2", value="copyright generative AI")
        output_csv = st.text_input("Output CSV", value=str(PROJECT_ROOT / "data" / "ai_copyright_dataset.csv"))
        if st.button("Scrape Dataset", use_container_width=True):
            try:
                with st.spinner("Collecting and enriching search results..."):
                    run_scrape(
                        queries=[query_1, query_2],
                        output_path=Path(output_csv),
                    )
                st.success("Dataset created. You can now rebuild the graph from the Build Graph tab.")
            except Exception as e:
                st.error(f"Error during scraping: {e}")


if __name__ == "__main__":
    main()
