import os
import json
import argparse
import subprocess
import time
import glob
import xml.etree.ElementTree as ET
import re
from tqdm import tqdm
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import PyPDF2

from grobid_client.grobid_client import GrobidClient
from ..utils.pdf_utils import FastPDFDownloader
from ..utils.string_utils import GrobidXMLParser


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return None


def extract_sections_from_grobid_xml(xml_content: str) -> Optional[str]:
    """
    Extract all sections from GROBID XML except results and conclusion sections.
    Returns the extracted text or None if extraction fails.
    """
    try:
        # Parse XML
        root = ET.fromstring(xml_content)
        
        # Define XML namespaces used by GROBID
        namespaces = {
            'tei': 'http://www.tei-c.org/ns/1.0'
        }
        
        # Find all sections
        sections = []
        
        # Extract title
        title_elem = root.find('.//tei:titleStmt/tei:title[@type="main"]', namespaces)
        if title_elem is not None and title_elem.text:
            sections.append(f"Title: {title_elem.text.strip()}")
        
        # Extract abstract
        abstract_elem = root.find('.//tei:abstract', namespaces)
        if abstract_elem is not None:
            abstract_text = ''.join(abstract_elem.itertext()).strip()
            if abstract_text:
                sections.append(f"Abstract: {abstract_text}")
        
        # Extract body sections
        body_elem = root.find('.//tei:body', namespaces)
        if body_elem is not None:
            # Find all div elements which typically represent sections
            divs = body_elem.findall('.//tei:div', namespaces)
            
            for div in divs:
                # Get section header
                head_elem = div.find('./tei:head', namespaces)
                if head_elem is not None:
                    section_title = head_elem.text.strip() if head_elem.text else ""
                    
                    # Skip results and conclusion sections
                    section_title_lower = section_title.lower()
                    if any(keyword in section_title_lower for keyword in 
                           ['result', 'conclusion', 'discussion', 'findings']):
                        continue
                    
                    # Extract section content
                    section_content = []
                    for elem in div:
                        if elem.tag.endswith('p'):  # paragraph
                            paragraph_text = ''.join(elem.itertext()).strip()
                            if paragraph_text:
                                section_content.append(paragraph_text)
                    
                    if section_content:
                        section_text = f"\n{section_title}:\n" + "\n\n".join(section_content)
                        sections.append(section_text)
        
        return "\n\n".join(sections) if sections else None
        
    except Exception as e:
        print(f"Error parsing GROBID XML: {e}")
        return None


def check_grobid_container():
    """Check if GROBID container is running and start if needed."""
    downloader = FastPDFDownloader()
    
    if not downloader.is_grobid_container_running():
        print("Starting GROBID container...")
        try:
            subprocess.run([
                "docker", "run", "-d", "--rm", "-p", "8070:8070", "lfoppiano/grobid:latest-crf"
            ], check=True)
            print("GROBID container started successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to start GROBID container: {e}")
            return False
    else:
        print("GROBID container is already running.")
    
    # Wait for container to be ready
    max_wait = 60  # seconds
    wait_time = 0
    while not downloader.is_grobid_container_running() and wait_time < max_wait:
        print("Waiting for GROBID container to be ready...")
        time.sleep(5)
        wait_time += 5
    
    if wait_time >= max_wait:
        print("GROBID container failed to start within timeout.")
        return False
    
    return True


def process_pdf_with_grobid(pdf_path: str, client: GrobidClient, output_dir: str) -> Dict[str, Any]:
    """Process a single PDF file with GROBID and extract sections excluding results/conclusion."""
    try:
        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]
        xml_output_path = os.path.join(output_dir, f"{pdf_name}.grobid.tei.xml")
        txt_output_path = os.path.join(output_dir, f"{pdf_name}.txt")
        
        result = {
            "pdf_file": pdf_filename,
            "txt_file": f"{pdf_name}.txt",
            "success": True,
            "method": "grobid"
        }
        
        # Check if XML already exists, otherwise process with GROBID
        xml_exists = os.path.exists(xml_output_path)
        
        if not xml_exists:
            # Create a temporary directory for this PDF
            temp_dir = os.path.join(output_dir, "temp_grobid")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Copy PDF to temp directory for GROBID processing
            temp_pdf_path = os.path.join(temp_dir, pdf_filename)
            subprocess.run(["cp", pdf_path, temp_pdf_path], check=True)
            
            # Process with GROBID
            grobid_result = client.process(
                "processFulltextDocument",
                temp_dir,
                n=1  # Single thread for this file
            )
            
            # Find the generated XML file in temp directory
            temp_xml_files = glob.glob(os.path.join(temp_dir, "*.grobid.tei.xml"))
            
            if temp_xml_files:
                # Move XML file to final output directory
                temp_xml_path = temp_xml_files[0]
                subprocess.run(["mv", temp_xml_path, xml_output_path], check=True)
                
                # Clean up temp PDF
                if os.path.exists(temp_pdf_path):
                    os.remove(temp_pdf_path)
                
                result["status"] = "processed"
            else:
                # GROBID failed, fall back to PyPDF2
                print(f"GROBID failed for {pdf_filename}, falling back to PyPDF2")
                fallback_text = extract_text_from_pdf(pdf_path)
                if fallback_text:
                    with open(txt_output_path, 'w', encoding='utf-8') as f:
                        f.write(fallback_text)
                    result["method"] = "pypdf2_fallback"
                    result["status"] = "fallback_success"
                    return result
                else:
                    return {
                        "pdf_file": pdf_filename,
                        "status": "both_methods_failed",
                        "error": "Both GROBID and PyPDF2 extraction failed",
                        "success": False
                    }
        else:
            result["status"] = "xml_already_exists"
        
        # Extract sections from XML if it exists
        if os.path.exists(xml_output_path):
            try:
                with open(xml_output_path, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
                
                extracted_text = extract_sections_from_grobid_xml(xml_content)
                
                if extracted_text:
                    with open(txt_output_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    result["status"] = "sections_extracted"
                else:
                    # XML parsing failed, fall back to PyPDF2
                    print(f"XML parsing failed for {pdf_filename}, falling back to PyPDF2")
                    fallback_text = extract_text_from_pdf(pdf_path)
                    if fallback_text:
                        with open(txt_output_path, 'w', encoding='utf-8') as f:
                            f.write(fallback_text)
                        result["method"] = "pypdf2_fallback"
                        result["status"] = "xml_parse_failed_fallback_success"
                    else:
                        return {
                            "pdf_file": pdf_filename,
                            "status": "both_methods_failed",
                            "error": "Both GROBID XML parsing and PyPDF2 extraction failed",
                            "success": False
                        }
                        
            except Exception as e:
                # XML processing failed, fall back to PyPDF2
                print(f"XML processing failed for {pdf_filename}: {e}, falling back to PyPDF2")
                fallback_text = extract_text_from_pdf(pdf_path)
                if fallback_text:
                    with open(txt_output_path, 'w', encoding='utf-8') as f:
                        f.write(fallback_text)
                    result["method"] = "pypdf2_fallback"
                    result["status"] = "xml_error_fallback_success"
                else:
                    return {
                        "pdf_file": pdf_filename,
                        "status": "both_methods_failed",
                        "error": f"XML processing error and PyPDF2 fallback failed: {e}",
                        "success": False
                    }
        
        return result
            
    except Exception as e:
        # Final fallback to PyPDF2 for any unexpected errors
        print(f"Unexpected error processing {pdf_filename}: {e}, trying PyPDF2 fallback")
        try:
            fallback_text = extract_text_from_pdf(pdf_path)
            if fallback_text:
                pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
                txt_output_path = os.path.join(output_dir, f"{pdf_name}.txt")
                with open(txt_output_path, 'w', encoding='utf-8') as f:
                    f.write(fallback_text)
                return {
                    "pdf_file": os.path.basename(pdf_path),
                    "txt_file": f"{pdf_name}.txt",
                    "method": "pypdf2_fallback",
                    "status": "exception_fallback_success",
                    "success": True
                }
        except Exception as fallback_e:
            pass
        
        return {
            "pdf_file": os.path.basename(pdf_path),
            "status": "error",
            "error": str(e),
            "success": False
        }


def process_pdf_direct_extraction(pdf_path: str, output_dir: str) -> Dict[str, Any]:
    """Process PDF using direct text extraction without GROBID."""
    try:
        pdf_filename = os.path.basename(pdf_path)
        pdf_name = os.path.splitext(pdf_filename)[0]
        txt_output_path = os.path.join(output_dir, f"{pdf_name}.txt")
        
        extracted_text = extract_text_from_pdf(pdf_path)
        
        if extracted_text:
            with open(txt_output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            return {
                "pdf_file": pdf_filename,
                "txt_file": f"{pdf_name}.txt",
                "method": "pypdf2_direct",
                "status": "direct_extraction_success",
                "success": True
            }
        else:
            return {
                "pdf_file": pdf_filename,
                "status": "direct_extraction_failed",
                "error": "PyPDF2 extraction failed",
                "success": False
            }
            
    except Exception as e:
        return {
            "pdf_file": os.path.basename(pdf_path),
            "status": "error",
            "error": str(e),
            "success": False
        }


def main():
    parser = argparse.ArgumentParser(description="Extract paper content from PDFs, excluding results and conclusion sections")
    parser.add_argument("--pdfs_dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--pdf_list", help="Optional JSON file with list of specific PDF filenames to process")
    parser.add_argument("--output_dir", help="Directory to save txt files (default: same as pdfs_dir)")
    parser.add_argument("--grobid_config", default="./GROBID_config.json", help="Path to GROBID config file")
    parser.add_argument("--progress_file", help="Optional path to write progress updates")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum number of concurrent processes (default: 4)")
    parser.add_argument("--skip_existing", action="store_true", help="Skip PDFs that already have txt files")
    parser.add_argument("--skip_grobid", action="store_true", help="Skip GROBID processing and use direct PDF extraction")
    args = parser.parse_args()

    # Set output directory
    if not args.output_dir:
        args.output_dir = args.pdfs_dir

    # Thread-safe progress tracking
    progress_lock = threading.Lock()
    
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
                with progress_lock:
                    with open(args.progress_file, "w") as f:
                        json.dump(progress_data, f)
            except Exception as e:
                print(f"Failed to write progress file: {e}")

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of PDF files to process
    if args.pdf_list:
        with open(args.pdf_list, 'r') as f:
            pdf_list_data = json.load(f)
            if isinstance(pdf_list_data, list):
                pdf_filenames = pdf_list_data
            elif isinstance(pdf_list_data, dict) and 'files' in pdf_list_data:
                pdf_filenames = pdf_list_data['files']
            else:
                raise ValueError("PDF list file must contain a list of filenames or have 'files' key")
        
        # Filter to only existing PDFs
        pdf_files = []
        for filename in pdf_filenames:
            pdf_path = os.path.join(args.pdfs_dir, filename)
            if os.path.exists(pdf_path):
                pdf_files.append(pdf_path)
            else:
                print(f"Warning: PDF file not found: {pdf_path}")
    else:
        # Get all PDF files from directory
        pdf_files = [
            os.path.join(args.pdfs_dir, f) 
            for f in os.listdir(args.pdfs_dir) 
            if f.endswith('.pdf')
        ]

    # Filter out PDFs that already have txt files if skip_existing is enabled
    if args.skip_existing:
        filtered_pdf_files = []
        for pdf_path in pdf_files:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            txt_path = os.path.join(args.output_dir, f"{pdf_name}.txt")
            if not os.path.exists(txt_path):
                filtered_pdf_files.append(pdf_path)
            else:
                print(f"Skipping {pdf_path} - txt already exists")
        pdf_files = filtered_pdf_files

    total_files = len(pdf_files)
    
    if total_files == 0:
        print("No PDF files to process.")
        return

    write_progress("starting", 0, total_files, "Initializing processing...")

    # Initialize GROBID client if not skipping GROBID
    client = None
    if not args.skip_grobid:
        # Check and start GROBID container
        if not check_grobid_container():
            print("Failed to start GROBID container. Falling back to direct PDF extraction.")
            args.skip_grobid = True
        else:
            client = GrobidClient(config_path=args.grobid_config)
    
    if args.skip_grobid:
        write_progress("processing", 0, total_files, "Starting direct PDF text extraction...")
    else:
        write_progress("processing", 0, total_files, "Starting PDF processing with GROBID...")
    
    processed_files = []
    failed_files = []
    completed_count = 0
    grobid_count = 0
    fallback_count = 0

    # Process PDFs with multithreading
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all processing tasks
        if args.skip_grobid:
            future_to_pdf = {
                executor.submit(process_pdf_direct_extraction, pdf_path, args.output_dir): pdf_path 
                for pdf_path in pdf_files
            }
        else:
            future_to_pdf = {
                executor.submit(process_pdf_with_grobid, pdf_path, client, args.output_dir): pdf_path 
                for pdf_path in pdf_files
            }
        
        # Process completed tasks with progress bar
        with tqdm(total=total_files, desc="Processing PDFs") as pbar:
            for future in as_completed(future_to_pdf):
                pdf_path = future_to_pdf[future]
                completed_count += 1
                
                try:
                    result = future.result()
                    if result.get('success'):
                        processed_files.append(result)
                        
                        # Count methods used
                        if result.get('method', '').startswith('pypdf2'):
                            fallback_count += 1
                        else:
                            grobid_count += 1
                            
                        print(f"Successfully processed: {result['pdf_file']} (method: {result.get('method', 'unknown')})")
                    else:
                        failed_files.append(result)
                        print(f"Failed to process: {result['pdf_file']} - {result.get('error', 'Unknown error')}")
                except Exception as e:
                    pdf_filename = os.path.basename(pdf_path)
                    error_result = {
                        "pdf_file": pdf_filename,
                        "status": "exception",
                        "error": str(e),
                        "success": False
                    }
                    failed_files.append(error_result)
                    print(f"Exception processing {pdf_filename}: {e}")
                
                # Update progress
                pbar.update(1)
                write_progress("processing", completed_count, total_files, 
                             f"Completed {completed_count}/{total_files} PDF processing")

    # Save processing results
    results = {
        "processed_files": processed_files,
        "failed_files": failed_files,
        "total_attempted": total_files,
        "successful": len(processed_files),
        "failed": len(failed_files),
        "grobid_processed": grobid_count,
        "fallback_processed": fallback_count,
        "skip_grobid": args.skip_grobid,
        "output_directory": args.output_dir
    }

    results_file = os.path.join(args.output_dir, "paper_content_processing_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    write_progress("completed", total_files, total_files, 
                  f"Processed {len(processed_files)}/{total_files} PDFs successfully")
    
    print(f"\nPaper content processing completed!")
    print(f"Successfully processed: {len(processed_files)}")
    print(f"Failed processing: {len(failed_files)}")
    if not args.skip_grobid:
        print(f"GROBID processed: {grobid_count}")
        print(f"Fallback processed: {fallback_count}")
    print(f"Text files saved to: {args.output_dir}")
    print(f"Results saved to: {results_file}")

    # Clean up temp directory if it exists
    temp_dir = os.path.join(args.output_dir, "temp_grobid")
    if os.path.exists(temp_dir):
        try:
            subprocess.run(["rm", "-rf", temp_dir], check=True)
        except subprocess.CalledProcessError:
            print(f"Warning: Could not clean up temporary directory: {temp_dir}")


if __name__ == '__main__':
    main()