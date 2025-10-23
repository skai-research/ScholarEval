# steps/snippet_search.py
import os, json
from tqdm import tqdm
import argparse
import time
import glob
import subprocess
import asyncio
from pathlib import Path
from grobid_client.grobid_client import GrobidClient
from ..utils.semantic_scholar import SemanticScholar
from ..utils.string_utils import GrobidXMLParser
from ..utils.pdf_utils import FastPDFDownloader


def main():
    parser = argparse.ArgumentParser(description="Search for snippets related to method.")
    parser.add_argument("--queries_file", required=True, help="Path to queries JSON file")
    parser.add_argument("--methods_file", required=True, help="Path to methods JSON file")
    parser.add_argument("--output_file", required=True, help="Path to output snippets JSON")
    parser.add_argument("--pdf_dir", default="./pdfs", help="Directory to store downloaded PDFs and XML files (default: ./pdfs)")
    parser.add_argument("--research_title", default="", help="Title of the research plan")
    parser.add_argument("--research_abstract", default="", help="Abstract of the research plan")
    parser.add_argument("--cutoff_date", help="Optional cutoff date (YYYY-MM-DD) for literature search.")
    parser.add_argument("--progress_file", help="Optional path to write progress updates for monitoring.")
    args = parser.parse_args()
    
    pdf_dir = Path(args.pdf_dir)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using PDF directory: {pdf_dir.absolute()}")
    
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
                print(f"Failed to write progress file: {e}")
    
    s2 = SemanticScholar(os.environ.get("S2_API_KEY"))
    downloader = FastPDFDownloader(pdf_dir=str(pdf_dir), email=os.environ.get("email"))
    
    if not downloader.is_grobid_container_running():
        subprocess.run(["docker", "run", "-d", "--rm", "-p", "8070:8070", "lfoppiano/grobid:latest-crf"])
    else:
        print("GROBID container is already running.")
    
    

    with open(args.queries_file) as f:
        queries = json.load(f)
    with open(args.methods_file) as f:
        clean_methods = json.load(f)['clean_methods']
    
    total_methods = len(clean_methods)
    write_progress("starting", 0, total_methods, "Initializing snippet search...")
    
    if args.cutoff_date:
        pub_date = args.cutoff_date
    elif args.research_title and args.research_abstract:
        eval_paper = s2.get_single_paper(args.research_title)
        found_paper = s2.is_right_paper(eval_paper['data'][0]['abstract'], args.research_abstract)
        pub_date = s2.get_safe_pub_date(eval_paper['data'][0]['publicationDate']) if found_paper else ""
    else:
        pub_date = ""

    references = {}
    total_snippets = 0
    
    snippet_search_start = time.time()
    
    if os.path.exists(args.output_file+'_references.json'):
        with open(args.output_file+'_references.json', 'r') as f:
            references = json.load(f)
        print(f"Loaded existing references from {args.output_file+'_references.json'}")
    else:
        print(f"Starting snippet search for {total_methods} methods...")
        for mid, method in enumerate(tqdm(clean_methods, desc='Finding Snippets for Each Method')):
            write_progress("processing", mid + 1, total_methods, f"Processing method {mid + 1}/{total_methods}: {method[:50]}...")
            try:
                print("Right before search_snippets")
                snippets = s2.search_snippets(queries["queries"][method], year=['', pub_date], limit=15)
                print(f"Right after search_snippets, got: {type(snippets)}")
                if snippets is None:
                    print("Warning: search_snippets returned None")
                    snippets = {}
                print(f"🔍 Method '{method[:30]}...': Retrieved snippets response: {type(snippets)}")
                if snippets:
                    print(f"  Snippets keys: {snippets.keys() if isinstance(snippets, dict) else 'Not a dict'}")
                    if isinstance(snippets, dict) and 'data' in snippets:
                        print(f"   Found {len(snippets['data'])} snippet items in data")
                    else:
                        print(f"    No 'data' key in snippets or snippets is not dict")
                else:
                    print(f"   No snippets returned (None/empty)")
                
                if snippets and 'data' in snippets:
                    all_paper_ids = s2.steal_cite_from_snip(snippets['data'])
                    references[method] = list(all_paper_ids)
                    total_snippets += len(snippets)
                    write_progress("processing", mid + 1, total_methods, f"Found {len(snippets)} snippets for this method. Total: {total_snippets} snippets")
                else:
                    print(f"Warning: No snippets found for method {method}.")
                    references[method] = []
            except Exception as e:
                print(f"Error processing method {method}: {e}")
                references[method] = []
    
    snippet_search_end = time.time()
    print(f" Snippet search took: {snippet_search_end - snippet_search_start:.2f} seconds")

    write_progress("saving", total_methods, total_methods, f"Saving {total_snippets} snippets to file...")
    with open(args.output_file+'_references.json', 'w') as f:
        json.dump(references, f, indent=4)
    
    # TIMING: Paper metadata collection
    metadata_start = time.time()
    
    # Now get metadata from those papers with S2
    unique_paper_ids = list(set(["CorpusId:" + ref for references in references.values() for ref in references]))
    print(f"Collecting metadata for {len(unique_paper_ids)} unique papers...")
    
    # Process in batches of 500
    batch_size = 500
    all_paper_metadata = []
    for i in range(0, len(unique_paper_ids), batch_size):
        batch_start = time.time()
        batch = unique_paper_ids[i:i + batch_size]
        print(f"   Processing batch {i//batch_size + 1}/{(len(unique_paper_ids) + batch_size - 1)//batch_size} ({len(batch)} papers)...")
        batch_metadata = s2.get_paper_bulk(batch)
        all_paper_metadata.extend(batch_metadata)
        batch_end = time.time()
        print(f"   ⏱️  Batch took: {batch_end - batch_start:.2f} seconds")
        
    # remove CorpusId: prefix
    unique_paper_ids = [upid.split(':')[1] for upid in unique_paper_ids]
    
    metadata_end = time.time()
    print(f" Paper metadata collection took: {metadata_end - metadata_start:.2f} seconds")
    write_progress("Downloading all pdfs to read", total_methods, total_methods, f"Reading through {len(unique_paper_ids)} pdfs...")
    # Download all pdfs
    print(f"should be {len(unique_paper_ids)} pdfs to download")
    
    # TIMING: PDF URL preparation
    url_prep_start = time.time()
    n_valid = 0
    pdf_data = []
    for paper_metadata, corpus_id in zip(all_paper_metadata, unique_paper_ids):
        if paper_metadata:
            pdf_info = paper_metadata.get('openAccessPdf', {})
            pdf_url = ''
            if pdf_info.get('url', None):
                pdf_url = pdf_info['url']
            elif pdf_info.get('disclaimer', None):
                pdf_url = downloader.extract_url(pdf_info['disclaimer'])
            if pdf_url:
                pdf_data.append((pdf_url, corpus_id))
            else:
                print(f"Could not find url for {corpus_id} -- {pdf_info['url']}")
        else:
            print(f"Could not find url for {corpus_id} -- {pdf_info['url']}")
    url_prep_end = time.time()
    print(f" PDF URL preparation took: {url_prep_end - url_prep_start:.2f} seconds")
    print(f"Found {len(pdf_data)} PDFs with valid URLs to attempt download")
    
    # TIMING: PDF downloads
    download_start = time.time()
    download_results = asyncio.run(downloader.download_pdfs_batch_async(pdf_data))
    download_end = time.time()
    print(f" PDF downloads took: {download_end - download_start:.2f} seconds for {len(pdf_data)} PDFs")
    
    # Debug: Check download results
    valid_count = 0
    failed_count = 0
    for i, result in enumerate(download_results):
        if result and not isinstance(result, Exception):
            valid_count += 1
            print(f"Download {i+1}: SUCCESS - {result}")
        else:
            failed_count += 1
            print(f"Download {i+1}: FAILED - {result}")
    
    print(f" Download Summary: {valid_count} successful, {failed_count} failed")
    valid = [result for result in download_results if result and not isinstance(result, Exception)]
    write_progress("Extracting paper details", total_methods, total_methods, f"{len(valid)} valid pdfs downloaded. Extracting paper details...")
    
    while not downloader.is_grobid_container_running():
        time.sleep(5)
    
    client = GrobidClient(config_path="./GROBID_config.json")
    status = s2.extract_sections_from_pdf(client, str(pdf_dir))
    # TIMING: XML parsing phase
    xml_parsing_start = time.time()
    extracted = {}
    xml_pattern = str(pdf_dir / "*.xml")
    xml_files = glob.glob(xml_pattern)
    print(f" Found {len(xml_files)} XML files to parse...")
    
    for i, xml_fp in enumerate(xml_files):
        xml_file_start = time.time()
        # Extract corpus ID from filename, working with any directory structure
        xml_path = Path(xml_fp)
        cid = xml_path.stem.split('.')[0]  # Get filename without extension, then split on '.' to get corpus ID
        print(f"Processing XML {i+1}/{len(xml_files)}: {cid}")
        
        with open(xml_fp, "r", encoding="utf-8") as file:
            xml_sample = file.read()
            parser = GrobidXMLParser(xml_sample)
            # Extract sections
            sections = parser.extract_sections()
            key_sections = ['method', 'task', 'baseline', 'methodology', 'approach', 'procedure', 'protocol', 'technique', 'design', 'framework', 'implementation', 'algorithm', 'process', 'workflow', 'strategy', 'preparation', 'synthesis', 'fabrication', 'construction','setup', 'apparatus', 'equipment', 'instrumentation', 'system', 'configuration', 'experimental', 'experiment', 'procedure', 'protocol', 'preparation', 'sample', 'specimen', 'material', 'device', 'platform', 'facility', 'laboratory', 'condition', 'parameter', 'result', 'results', 'finding', 'findings', 'outcome', 'observation', 'data', 'measurement', 'performance', 'evaluation', 'validation', 'testing', 'characterization', 'analysis', 'assessment', 'output', 'response', 'behavior', 'effect', 'impact']
            paper = ""
            for section in sections:
                if any([x in section['header'].lower() for x in key_sections]) and section['section_number']:
                    paper += f"# {section['header']}\n\t"
                    paper += '\n\t'.join(section['paragraphs']) + '\n'
            extracted[cid] = {
                'n_char': len(paper),
                'n_words': len(paper.split()),
                'sections_used': [section['header'] for section in sections if any([x in section['header'].lower() for x in key_sections])],
                'sections_all': [section['header'] for section in sections],
                'paper': paper,
            }
        
        xml_file_end = time.time()
        print(f"  XML file {cid} took: {xml_file_end - xml_file_start:.2f} seconds")
    
    xml_parsing_end = time.time()
    total_xml_time = xml_parsing_end - xml_parsing_start
    print(f" XML parsing took: {total_xml_time:.2f} seconds total ({total_xml_time/len(xml_files):.2f}s per file average)")
    save_start = time.time()
    with open(args.output_file+'_papers.json', 'w') as f:
        json.dump(extracted, f, indent=4)
    save_end = time.time()
    print(f" Final JSON save took: {save_end - save_start:.2f} seconds")


if __name__ == '__main__':
    main()
