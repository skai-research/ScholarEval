#!/usr/bin/env python3
import argparse
import json
import logging
import time
import os
import glob
import subprocess
import asyncio
from datetime import datetime
from grobid_client.grobid_client import GrobidClient
from ..utils.semantic_scholar import SemanticScholar
from ..utils.string_utils import GrobidXMLParser
from ..utils.pdf_utils import FastPDFDownloader

def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def is_date_after(cutoff: str, paper: str) -> bool:
    if not cutoff or not paper:
        return True
    cutoff_date = datetime.strptime(cutoff, "%Y-%m-%d")
    paper_date = datetime.strptime(paper, "%Y-%m-%d")
    return cutoff_date > paper_date

async def main():
    setup_logger()
    parser = argparse.ArgumentParser()
    parser.add_argument("--relevant_papers", required=True, help="JSON of relevant papers")
    parser.add_argument("--output_file", required=True, help="Output JSON file")
    parser.add_argument("--rec_limit", type=int, default=8, help="Number of recs per paper")
    parser.add_argument("--sleep_between_calls", type=float, default=1,
                        help="Seconds to sleep between API calls")
    parser.add_argument("--pdf_dir", default="./pdfs", help="Directory to store downloaded PDFs")
    parser.add_argument("--grobid_config", default="./GROBID_config.json", help="GROBID config file path")
    parser.add_argument("--augmentation_type", choices=["related_work", "all"], default="all",
                        help="Type of augmentation: 'related_work' (recommendations + related work sections) or 'all' (recommendations + all references)")
    parser.add_argument("--cutoff_date", help="Optional cutoff date (YYYY-MM-DD) for literature search.")
    parser.add_argument("--max_refs_per_paper", type=int, default=10,
                        help="Max number of references to fetch per paper (used with 'all' augmentation type)")
    args = parser.parse_args()

    s2 = SemanticScholar(api_key=os.environ.get("S2_API_KEY_1"))
    downloader = FastPDFDownloader()
    
    # Start GROBID container if not running
    if not downloader.is_grobid_container_running():
        logging.info("Starting GROBID container...")
        subprocess.run(["docker", "run", "-d", "--rm", "-p", "8070:8070", "lfoppiano/grobid:latest-crf"])
        # Wait for container to be ready
        while not downloader.is_grobid_container_running():
            logging.info("Waiting for GROBID container to start...")
            time.sleep(5)
    else:
        logging.info("GROBID container is already running.")
    
    papers_data = json.load(open(args.relevant_papers))
    papers = papers_data['papers'] if 'papers' in papers_data else papers_data
    augmented = {p['paperId']: p for p in papers}
    initial_paper_count = len(papers)
    
    # Tracking counters
    recommendations_added = 0
    related_work_added = 0
    references_added = 0  # For simple references API
    pdfs_downloaded = 0
    xml_files_processed = 0

    # Filter papers with relevance score >= 3
    high_relevance_papers = [p for p in papers if p.get('relevance_score', 0) >= 3]
    logging.info(f"Starting with {initial_paper_count} papers")
    logging.info(f"Processing {len(high_relevance_papers)} papers with relevance score >= 3")
    logging.info(f"Augmentation type: {args.augmentation_type}")

    for p in high_relevance_papers:
        pid = p['paperId']
        relevance_score = p.get('relevance_score', 0)
        logging.info(f"Processing paper {pid} (relevance score: {relevance_score})")

        logging.info(f" → Fetching recommendations (limit={args.rec_limit})")
        recommendations_for_this_paper = 0
        try:
            recs = s2.get_recommendations_multi_seed([pid], limit=args.rec_limit)
            for r in recs:
                rid = r['paperId']
                logging.info(f"    • Recommendation: {rid}")
                if rid not in augmented:
                    paper_details = s2.get_paper_details(rid)
                    if (is_date_after(args.cutoff_date, paper_details.get('publicationDate', '')) if args.cutoff_date else True):
                        augmented[rid] = paper_details
                        recommendations_added += 1
                        recommendations_for_this_paper += 1
            logging.info(f"   Added {recommendations_for_this_paper} new recommendations for {pid}")
        except Exception as e:
            logging.warning(f"Failed to fetch recommendations for {pid}: {e}")
        time.sleep(args.sleep_between_calls)

        if args.augmentation_type == "related_work":
            # Extract from related work sections in PDFs
            logging.info(f" → Processing related work section for {pid}")
        else:  # augmentation_type == "all"
            # Use simple references API (from old code)
            logging.info(f" → Fetching all references (max={args.max_refs_per_paper or 'all'})")
        try:
            if args.augmentation_type == "related_work":
                # Get paper metadata to find PDF
                paper_details = s2.get_paper_details(pid)
                pdf_url = None
                
                if paper_details.get('openAccessPdf'):
                    pdf_info = paper_details['openAccessPdf']
                    if pdf_info.get('url'):
                        pdf_url = pdf_info['url']
                    elif pdf_info.get('disclaimer'):
                        pdf_url = downloader.extract_url(pdf_info['disclaimer'])
                
                if not pdf_url:
                    logging.warning(f"No PDF available for {pid}, skipping related work extraction")
                    continue
                    
                # Download PDF
                corpus_id = str(paper_details.get('corpusId', pid))
                pdf_path = await download_single_pdf(downloader, pdf_url, corpus_id, args.pdf_dir)
                
                if not pdf_path:
                    logging.warning(f"Failed to download PDF for {pid}")
                    continue
                    
                # Process with GROBID (defer to batch processing after all downloads)
                pdfs_downloaded += 1
                logging.info(f"PDF downloaded successfully for {pid} (total PDFs: {pdfs_downloaded})")
                
            else:  # augmentation_type == "all"
                # Use simple references API (from old code)
                refs = s2.get_references(pid, max_references=args.max_refs_per_paper)
                references_for_this_paper = 0
                for cited in refs:
                    cid = cited.get("paperId")
                    if not cid:
                        continue
                    logging.info(f"    • Reference: {cid}")
                    if cid not in augmented:
                        details = s2.get_paper_details(cid)
                        augmented[cid] = details
                        references_added += 1
                        references_for_this_paper += 1
                logging.info(f"   Added {references_for_this_paper} new references for {pid}")
                
        except Exception as e:
            if args.augmentation_type == "related_work":
                logging.warning(f"Failed to process related work for {pid}: {e}")
            else:
                logging.warning(f"Failed to fetch references for {pid}: {e}")
        
        time.sleep(args.sleep_between_calls)

    # Process all downloaded PDFs with GROBID in batch (only for related_work augmentation)
    if args.augmentation_type == "related_work" and pdfs_downloaded > 0:
        logging.info(f"Processing {pdfs_downloaded} downloaded PDFs with GROBID...")
        try:
            client = GrobidClient(config_path=args.grobid_config)
            # Wait for GROBID to be ready
            while not is_grobid_ready():
                logging.info("Waiting for GROBID server to be ready...")
                time.sleep(5)
            
            # Process all PDFs in the directory
            result = s2.extract_sections_from_pdf(client, args.pdf_dir)
            logging.info(f"GROBID processing completed")
            
            # Now extract related work from all XML files
            pdf_files = glob.glob(os.path.join(args.pdf_dir, "*.pdf"))
            xml_files = glob.glob(os.path.join(args.pdf_dir, "*.grobid.tei.xml"))
            logging.info(f"Found {len(pdf_files)} PDF files to process for related work extraction")
            logging.info(f"Found {len(xml_files)} GROBID XML files (.grobid.tei.xml)")
            
            for pdf_path in pdf_files:
                # GROBID creates .grobid.tei.xml files, not .xml
                xml_path = pdf_path.replace('.pdf', '.grobid.tei.xml')
                if os.path.exists(xml_path):
                    xml_files_processed += 1
                    try:
                        corpus_id = os.path.basename(pdf_path).replace('.pdf', '')
                        logging.info(f"[{xml_files_processed}/{len(pdf_files)}] Extracting related work from {corpus_id}")
                        
                        logging.info(f"   Calling extract_related_work_references for {corpus_id}")
                        related_work_refs = extract_related_work_references(xml_path, s2)
                        logging.info(f"   extract_related_work_references returned {len(related_work_refs) if related_work_refs else 0} papers")
                        
                        # Add related work papers to augmented list
                        new_papers_from_this_xml = 0
                        for ref_paper in related_work_refs:
                            ref_id = ref_paper.get('paperId')
                            if ref_id and ref_id not in augmented:
                                logging.info(f"    • Added related work paper: {ref_id}")
                                augmented[ref_id] = ref_paper
                                related_work_added += 1
                                new_papers_from_this_xml += 1
                            elif ref_id:
                                logging.info(f"    • Skipped duplicate paper: {ref_id}")
                                
                        logging.info(f"   Found {len(related_work_refs)} total references, added {new_papers_from_this_xml} new papers from {corpus_id}")
                        
                    except Exception as e:
                        logging.warning(f"Failed to extract related work from {xml_path}: {e}")
                else:
                    logging.warning(f"XML file not found for {pdf_path}")
            
            logging.info(f"Processed {xml_files_processed} XML files out of {len(pdf_files)} PDFs")
            
        except Exception as e:
            logging.error(f"GROBID batch processing failed: {e}")
    else:
        if args.augmentation_type == "related_work":
            logging.info("No PDFs downloaded for related work extraction")
        else:
            logging.info("Skipping GROBID processing for 'all' augmentation type")

    # Save augmented results
    out = list(augmented.values())
    final_paper_count = len(out)
    
    with open(args.output_file, "w") as f:
        json.dump(out, f, indent=2)
    
    # Summary statistics
    logging.info("=" * 60)
    logging.info("AUGMENTATION SUMMARY:")
    logging.info(f"Initial papers: {initial_paper_count}")
    logging.info(f"Papers processed (relevance >= 3): {len(high_relevance_papers)}")
    logging.info(f"PDFs downloaded: {pdfs_downloaded}")
    logging.info(f"XML files processed: {xml_files_processed}")
    logging.info(f"Papers added via recommendations: {recommendations_added}")
    if args.augmentation_type == "related_work":
        logging.info(f"Papers added via related work: {related_work_added}")
        total_added = recommendations_added + related_work_added
    else:
        logging.info(f"Papers added via references: {references_added}")
        total_added = recommendations_added + references_added
    logging.info(f"Total papers added: {total_added}")
    logging.info(f"Final paper count: {final_paper_count}")
    logging.info(f"Growth factor: {final_paper_count/initial_paper_count:.2f}x")
    logging.info("=" * 60)
    logging.info(f"Saved {final_paper_count} papers to {args.output_file}")

async def download_single_pdf(downloader, pdf_url, corpus_id, pdf_dir):
    """Download a single PDF and return the file path."""
    try:
        # Use the async downloader for a single PDF
        pdf_data = [(pdf_url, corpus_id)]
        results = await downloader.download_pdfs_batch_async(pdf_data, save_dir=pdf_dir)
        
        if results and len(results) > 0:
            return os.path.join(pdf_dir, f"{corpus_id}.pdf")
        return None
    except Exception as e:
        logging.error(f"Error downloading PDF {corpus_id}: {e}")
        return None

def is_grobid_ready():
    """Check if GROBID server is ready."""
    try:
        import requests
        response = requests.get("http://localhost:8070/api/isalive", timeout=5)
        return response.status_code == 200
    except:
        return False

def extract_related_work_references(xml_path, s2_client):
    """Extract papers referenced in the related work section."""
    try:
        with open(xml_path, "r", encoding="utf-8") as file:
            xml_content = file.read()
            
        parser = GrobidXMLParser(xml_content)
        
        # Find related work section
        related_work_section = parser.find_related_work_section()
        
        if not related_work_section:
            logging.info("   No related work section found")
            return []
        
        logging.info(f"   Found related work section: '{related_work_section['header']}'")
        logging.info(f"   Related work section length: {len(related_work_section['full_text'])} characters")
        
        # Get bibliography entries
        bibliography = parser.extract_bibliography()
        
        if not bibliography:
            logging.info("   No bibliography found")
            return []
        
        # Extract reference citations from related work text
        section_refs = parser.extract_references_from_section(related_work_section['full_text'])
        
        logging.info(f"   Found {len(bibliography)} total bibliography entries")
        logging.info(f"   Found {len(section_refs)} reference citations in related work section")
        
        # Filter bibliography to only entries mentioned in related work
        relevant_bib_entries = []
        for bib_entry in bibliography:
            if any(ref_id in section_refs for ref_id in [bib_entry.get('xml:id', ''), bib_entry.get('id', '')]):
                relevant_bib_entries.append(bib_entry)
        
        logging.info(f"   {len(relevant_bib_entries)} bibliography entries are referenced in related work section")
        
        # Search for papers in bibliography using Semantic Scholar
        found_papers = []
        search_limit = min(20, len(relevant_bib_entries))  # Use relevant entries, limit to 20
        entries_to_search = relevant_bib_entries if relevant_bib_entries else bibliography[:20]
        
        logging.info(f"   Searching for {search_limit} papers from {'relevant' if relevant_bib_entries else 'first 20'} bibliography entries...")
        
        for i, bib_entry in enumerate(entries_to_search[:search_limit]):
            if not bib_entry.get('title') or not bib_entry.get('authors'):
                continue
                
            title = bib_entry['title'].strip()
            authors = bib_entry['authors']
            
            if len(title) < 10:  # Skip very short titles
                continue
                
            logging.info(f"   [{i+1}/{search_limit}] Searching for: {title[:50]}...")
            
            try:
                # Search using title and first author
                query = title
                if authors and len(authors) > 0:
                    first_author = authors[0].split()[-1]  # Get last name
                    query = f"{title} {first_author}"
                
                # Use the existing search method
                search_results = s2_client.search_top_papers(query, limit=3)
                
                if search_results:
                    # Find best match based on title similarity
                    best_match = find_best_title_match(title, search_results)
                    
                    if best_match:
                        # Get full paper details
                        paper_details = s2_client.get_paper_details(best_match['paperId'])
                        found_papers.append(paper_details)
                        logging.info(f"     ✓ Found: {best_match['title'][:50]}...")
                    else:
                        logging.info(f"     ✗ No good match found (similarity too low)")
                else:
                    logging.info(f"     ✗ No search results from Semantic Scholar")
                    
            except Exception as e:
                logging.warning(f"     Error searching for paper '{title[:30]}...': {e}")
                continue
            
            # Rate limiting
            time.sleep(1)
            
        logging.info(f"   Successfully found {len(found_papers)} papers from related work section")
        
        return found_papers
        
    except Exception as e:
        logging.error(f"Error extracting related work references: {e}")
        return []

def find_best_title_match(target_title, search_results):
    """Find the best matching paper based on title similarity."""
    target_words = set(target_title.lower().split())
    best_match = None
    best_score = 0
    
    for paper in search_results:
        if not paper.get('title'):
            continue
            
        paper_title = paper['title'].lower()
        paper_words = set(paper_title.split())
        
        intersection = len(target_words & paper_words)
        union = len(target_words | paper_words)
        
        if union > 0:
            similarity = intersection / union
            
            substring_bonus = 0
            if target_title.lower() in paper_title or paper_title in target_title.lower():
                substring_bonus = 0.2
            
            total_score = similarity + substring_bonus
            
            if total_score > best_score and total_score > 0.5: 
                best_score = total_score
                best_match = paper
    
    return best_match

if __name__ == "__main__":
    asyncio.run(main())