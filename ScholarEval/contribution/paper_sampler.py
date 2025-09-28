#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Filter papers by relevance_score >= 3, sort by score in decreasing order, and take top 25 papers"
    )
    parser.add_argument("--input_file", help="Path to input JSON file")
    parser.add_argument("--output_file", help="Path to output JSON file")
    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist")
        sys.exit(1)

    # Load papers from input file
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            papers = json.load(f)["papers"]
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON from '{args.input_file}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to read '{args.input_file}': {e}")
        sys.exit(1)

    # Validate that papers is a list
    if not isinstance(papers, list):
        print(f"Error: Expected JSON array, got {type(papers).__name__}")
        sys.exit(1)

    num_papers = len(papers)
    print(f"Found {num_papers} papers in input file")
    
    # Filter papers by relevance_score >= 3
    relevant_papers = [paper for paper in papers if paper.get('relevance_score', 0) >= 3]
    num_relevant = len(relevant_papers)
    print(f"Found {num_relevant} papers with relevance_score >= 3")
    
    # Sort filtered papers by relevance_score in decreasing order
    sorted_papers = sorted(relevant_papers, key=lambda x: x.get('relevance_score', 0), reverse=True)
    print(f"Sorted {num_relevant} papers by relevance_score in decreasing order")

    # Take top 25 papers
    top_papers = sorted_papers[:25]
    print(f"Selected top {len(top_papers)} papers from filtered set (max 25)")

    # Ensure output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save top papers to output file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(top_papers, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {len(top_papers)} papers to '{args.output_file}'")
    except Exception as e:
        print(f"Error: Failed to write to '{args.output_file}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()