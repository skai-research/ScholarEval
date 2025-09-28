import argparse
import json
import logging
import os
import re
from typing import List, Dict
from tqdm import tqdm
from ..engine.litellm_engine import LLMEngine


def extract_url_from_disclaimer(disclaimer: str) -> str:
    # Regex pattern to match http(s) URLs
    print(f"Extracting URL from disclaimer: {disclaimer}")
    url_pattern = r"https?://[^\s,]+"
    matches = re.findall(url_pattern, disclaimer)
    
    if not matches:
        return "no_url"
    
    # Return first URL unless it contains "unpaywall", then return second URL
    if "unpaywall" in matches[0]:
        return matches[1] if len(matches) > 1 else matches[0]
    else:
        return matches[0]

def format_reference(paper_metadata) -> str:
    """
    Format the reference string for a paper.
    """
    if paper_metadata is None:
        return "Unknown. No Title. No Year. No Venue. no_url"
    
    # Handle title
    title = paper_metadata.get("paper_title", "No Title")
    if title is None:
        title = "No Title"
    
    # Handle authors
    paper_authors = paper_metadata.get("paper_authors", [])
    if paper_authors is None:
        paper_authors = []
    authors = ", ".join(author.get("name", "Unknown") for author in paper_authors if isinstance(author, dict))
    if not authors:
        authors = "Unknown"
    
    # Handle publication date
    publication_date = paper_metadata.get("publication_date", "No Date")
    if publication_date is None:
        publication_date = "No Date"
    publication_year = publication_date.split("-")[0] if publication_date != "No Date" else "No Year"
    
    publication_venue = paper_metadata.get("paper_venue", "No Venue")
    if publication_venue is None:
        publication_venue = "No Venue"

    # Handle PDF URL
    paper_pdf = paper_metadata.get("paper_pdf", {})
    if paper_pdf is None:
        paper_pdf = {}
    disclaimer = paper_pdf.get("disclaimer", "no_url")
    if disclaimer is None:
        disclaimer = "no_url"
    url = extract_url_from_disclaimer(disclaimer)
    
    return f"{authors}. {title}. {publication_year}. {publication_venue}. {url}"

def main(): 
    parser = argparse.ArgumentParser(description="Prepare final contribution context for papers.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input paper metadata file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output context file.")
    args = parser.parse_args()

    # Load papers from input file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file {args.input_file} does not exist.")
    
    with open(args.input_file, 'r') as f:
        try:
            papers = json.load(f)["comparisons"]
            if not isinstance(papers, list):
                raise ValueError("Input file must contain a JSON array of papers.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from input file: {e}")

    context = []
    for paper in tqdm(papers, desc="Processing papers"):
        paper_reference = format_reference(paper)
        context.append({
            "paperReference": paper_reference,
            "comparison": paper.get("comparison")
        })
    # Write context to output file
    with open(args.output_file, 'w') as f:
        json.dump(context, f, indent=4)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    main()
