import argparse
import json
import logging
import time
from typing import Dict, Any, Set
import os

from ..utils.semantic_scholar import SemanticScholar

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Extract top papers and influential references.")
    parser.add_argument("--queries_file", required=True, help="Path to JSON file mapping contributions to queries.")
    parser.add_argument("--output_file", required=True, help="Path to save output metadata as JSON.")
    parser.add_argument("--cutoff_date", help="Optional cutoff date (YYYY-MM-DD) for literature search.")
    parser.add_argument("--progress_file", help="Optional path to write progress updates for monitoring.")
    args = parser.parse_args()

    def write_progress(status: str, current: int = 0, total: int = 0, details: str = ""):
        """Write progress update to file for monitoring by external processes."""
        if args.progress_file:
            progress_data = {
                "status": status,
                "current": current,
                "total": total,
                "details": details,
                "timestamp": time.time()
            }
            try:
                with open(args.progress_file, "w") as f:
                    json.dump(progress_data, f)
            except Exception as e:
                logging.warning(f"Failed to write progress file: {e}")

    logging.info("Initializing Semantic Scholar client")
    s2 = SemanticScholar(os.environ.get("S2_API_KEY"))

    logging.info("Loading queries from %s", args.queries_file)
    with open(args.queries_file, "r", encoding="utf-8") as f:
        query_map = json.load(f)["queries"]

    # Calculate total queries for progress tracking
    total_queries = sum(len(queries) for queries in query_map.values())
    current_query = 0
    write_progress("starting", 0, total_queries, "Initializing paper search...")

    all_papers: Dict[str, Dict[str, Any]] = {}
    seen: Set[str] = set()

    for contrib, queries in query_map.items():
        logging.info("Processing contribution: %s", contrib)
        write_progress("processing", current_query, total_queries, f"Processing contribution: {contrib}")
        
        for query in queries:
            current_query += 1
            logging.info("  Running query: %s", query)
            write_progress("querying", current_query, total_queries, f"Query {current_query}/{total_queries}: {query[:50]}...")
            
            try:
                top_papers = s2.search_top_papers(query, max_date=args.cutoff_date)
            except Exception as e:
                logging.error("  Failed query '%s': %s", query, e)
                continue

            for paper in top_papers:
                pid = paper["paperId"]
                if pid in seen:
                    continue
                seen.add(pid)

                logging.info("    Storing top paper: %s", pid)
                all_papers[pid] = s2.extract_metadata(paper)

            write_progress("querying", current_query, total_queries, f"Found {len(all_papers)} unique papers so far...")
            time.sleep(1)

    write_progress("saving", total_queries, total_queries, f"Saving {len(all_papers)} papers to file...")
    logging.info("Writing %d papers to %s", len(all_papers), args.output_file)
    with open(args.output_file, "w", encoding="utf-8") as out:
        json.dump(list(all_papers.values()), out, indent=2, ensure_ascii=False)

    write_progress("completed", total_queries, total_queries, f"Successfully extracted {len(all_papers)} papers")
    logging.info("Done.")

if __name__ == "__main__":
    main()
