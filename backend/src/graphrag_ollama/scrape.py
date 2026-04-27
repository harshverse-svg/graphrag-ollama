from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import trafilatura
from serpapi import GoogleSearch
from youtube_transcript_api import YouTubeTranscriptApi

from .config import load_runtime_config


def collect_search_results(
    queries: list[str],
    num_results: int = 10,
) -> tuple[pd.DataFrame, list[dict]]:
    config = load_runtime_config()
    if not config.serpapi_key:
        raise ValueError("SERPAPI_KEY is missing. Add it to your .env file before scraping.")

    article_rows: list[dict] = []
    video_rows: list[dict] = []
    raw_results: list[dict] = []

    for query in queries:
        print(f"Searching: {query}")
        params = {
            "engine": "google",
            "q": query,
            "google_domain": "google.com",
            "hl": "en",
            "gl": "us",
            "api_key": config.serpapi_key,
        }

        results = GoogleSearch(params).get_dict()
        raw_results.append({"query": query, "results": results})

        for article in results.get("organic_results", [])[:num_results]:
            article_rows.append(
                {
                    "query": query,
                    "title": article.get("title"),
                    "snippet": article.get("snippet"),
                    "source": article.get("source"),
                    "date": article.get("date"),
                    "url": article.get("link"),
                    "type": "article",
                }
            )

        for video in results.get("inline_videos", []):
            video_rows.append(
                {
                    "query": query,
                    "title": video.get("title"),
                    "snippet": None,
                    "source": video.get("channel"),
                    "date": None,
                    "url": video.get("link"),
                    "type": "video",
                }
            )

    dataframe = pd.DataFrame(article_rows + video_rows)
    if dataframe.empty:
        return dataframe, raw_results

    dataframe = dataframe.drop_duplicates(subset="url").reset_index(drop=True)
    return dataframe, raw_results


def _scrape_url(url: str) -> dict:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return {"url": url, "full_text": None, "status": "failed_download"}

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_precision=False,
            deduplicate=True,
        )
        return {
            "url": url,
            "full_text": text.strip() if text else None,
            "status": "success" if text else "failed_extraction",
        }
    except Exception as exc:
        return {"url": url, "full_text": None, "status": f"error: {exc}"}


def _get_youtube_id(url: str) -> str | None:
    patterns = [
        r"youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
        r"youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
        r"youtu\.be/([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _get_transcript(video_id: str) -> str | None:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=["en", "en-US", "en-GB"],
        )
    except Exception:
        return None

    return " ".join(item["text"] for item in transcript)


def enrich_search_results(
    dataframe: pd.DataFrame,
    max_workers: int = 5,
    delay: float = 1.0,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    dataframe = dataframe.copy()
    dataframe["full_text"] = None
    dataframe["video_id"] = None
    dataframe["status"] = "pending"

    article_mask = dataframe["type"] == "article"
    video_mask = dataframe["type"] == "video"

    article_df = dataframe[article_mask].copy()
    video_df = dataframe[video_mask].copy()

    if not article_df.empty:
        print(f"Scraping {len(article_df)} article URLs")
        url_to_result: dict[str, dict] = {}

        def fetch_with_delay(url: str) -> dict:
            time.sleep(delay)
            return _scrape_url(url)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_with_delay, row["url"]): idx
                for idx, row in article_df.iterrows()
            }
            for future in as_completed(futures):
                result = future.result()
                url_to_result[result["url"]] = result

        for idx, row in article_df.iterrows():
            result = url_to_result.get(row["url"], {})
            dataframe.at[idx, "full_text"] = result.get("full_text")
            dataframe.at[idx, "status"] = result.get("status", "unknown")

    if not video_df.empty:
        print(f"Fetching transcripts for {len(video_df)} videos")
        for idx, row in video_df.iterrows():
            video_id = _get_youtube_id(row["url"])
            transcript = _get_transcript(video_id) if video_id else None
            dataframe.at[idx, "video_id"] = video_id
            dataframe.at[idx, "full_text"] = transcript
            dataframe.at[idx, "status"] = "success" if transcript else "no_transcript"

    return dataframe


def run_scrape(
    queries: list[str],
    output_path: Path,
    num_results: int = 10,
    max_workers: int = 5,
    delay: float = 1.0,
) -> pd.DataFrame:
    dataframe, _ = collect_search_results(queries=queries, num_results=num_results)
    enriched = enrich_search_results(dataframe, max_workers=max_workers, delay=delay)
    clean = enriched[enriched["status"] == "success"].reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")
    print(f"Rows kept: {len(clean)}")
    return clean
