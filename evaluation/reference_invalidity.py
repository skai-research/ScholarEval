#!/usr/bin/env python3
import os
import sys
import argparse
import re
import time
import requests
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
import statistics
import logging
from datetime import datetime

# -------- DOI + bot-wall helpers --------

DOI_RE = re.compile(r'10\.\d{4,9}/[-._;()/:A-Z0-9]+', re.I)

def extract_doi_from_url(url: str):
    m = DOI_RE.search(url)
    return m.group(0) if m else None

def is_bot_challenge(content: str, headers: dict, url: str) -> bool:
    """Detect common anti-bot interstitials (Cloudflare/Akamai/PerimeterX)."""
    cl = content.lower() if isinstance(content, str) else ""
    hkeys = {k.lower(): str(v).lower() for k, v in (headers or {}).items()}

    signals = [
        'just a moment',                 # Cloudflare interstitial title
        'enable javascript and cookies', # challenge page body text
        '__cf_chl_', 'cf-ray', 'cf_chl', # CF tokens
        'access denied',
        'are you a robot',
        'perimeterx',
        'akamai bot manager',
        'request unsuccessful',
        'temporarily blocked',
        'bot detected',
    ]
    if any(s in cl for s in signals):
        return True
    if 'server' in hkeys and 'cloudflare' in hkeys['server']:
        return True
    # Many bot walls add a noindex robots meta
    if 'noindex,nofollow' in cl or 'meta name="robots" content="noindex' in cl:
        return True
    return False

def resolve_doi_exists(doi: str, timeout=10) -> bool:
    """
    Resolve a DOI via doi.org as canonical existence check.
    For hallucination detection we choose strict behavior:
    - 404/410/>=400 or bot wall => False
    """
    try:
        r = requests.get(
            f'https://doi.org/{doi}',
            timeout=timeout,
            allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        if r.status_code in (404, 410):
            return False
        if r.status_code >= 400:
            return False
        if is_bot_challenge(r.text, r.headers, r.url):
            return False
        return True
    except requests.RequestException:
        return False

# -------- existing helpers (hardened) --------

def extract_links(text):
    """Extract all HTTP/HTTPS links from text using regex."""
    url_pattern = r'https?://[^\s\)\]\},"\'<>]+'
    links = re.findall(url_pattern, text, re.IGNORECASE)

    cleaned_links = []
    for link in links:
        link = re.sub(r'[.,;:!?\)\]\}]+$', '', link)
        if link:
            cleaned_links.append(link)

    unique_links = list(dict.fromkeys(cleaned_links))
    return unique_links


def check_content_for_not_found(content, url):
    """Check if the content indicates a 'not found' or similar error page (or bot wall)."""
    content_lower = content.lower()

    # Treat bot interstitials as "not valid content"
    if is_bot_challenge(content, {}, url):
        return True

    not_found_patterns = [
        'page not found',
        'page cannot be found',
        'page does not exist',
        'page was not found',
        'page you were looking for does not exist',
        'page you requested does not exist',
        'the page you are looking for does not exist',
        'the page you were looking for could not be found',
        '404 error',
        'error 404',
        'http 404',
        'not found error',
        'file not found',
        'document not found',
        'resource not found',
        'content not found',
        'article not found',
        'page has been removed',
        'page has been deleted',
        'page no longer exists',
        'page is no longer available',
        'this page is missing',
        'sorry, page not found',
        'oops! page not found',
        'we could not find the page',
        'the requested page could not be found',
        'the requested url was not found',
        'the page has moved or been deleted',
        'page unavailable',
        'content unavailable',
        'page temporarily unavailable',
        'temporarily unavailable',
        'page expired',
        'link is broken',
        'broken link',
        'invalid link',
        'dead link',
        'the link you followed may be broken',
    ]

    domain = urlparse(url).netloc.lower()
    site_specific_patterns = {
        'onlinelibrary.wiley.com': ['error 404', 'page not found'],
        'sciencedirect.com': ['page not found', 'report missing page'],
        'springer.com': ['page not found', 'the page you requested does not exist', 'page cannot be displayed'],
        'nature.com': ['page not found', 'page does not exist', 'article not found'],
        'ieee.org': ['document not found', 'page not found', 'invalid document'],
        'acm.org': ['page not found', 'document not available', 'page does not exist'],
        'arxiv.org': ['submission not found', 'paper not found', 'invalid paper id'],
        'pubmed.ncbi.nlm.nih.gov': ['page not found', 'pmid not found', 'article not found'],
        'researchgate.net': ['page not found', 'publication not found', 'content not available'],
    }

    for pattern in not_found_patterns:
        if pattern in content_lower:
            return True

    for site_domain, patterns in site_specific_patterns.items():
        if site_domain in domain:
            if any(p in content_lower for p in patterns):
                return True

    # Strip tags and compress whitespace
    text_content = re.sub(r'<[^>]+>', '', content)
    text_content = re.sub(r'\s+', ' ', text_content).strip()

    if len(text_content) < 200:
        error_keywords = ['error', 'not found', 'missing', 'unavailable', 'broken']
        if any(keyword in content_lower for keyword in error_keywords):
            return True

    return False


def check_link_validity(url, timeout=10):
    """
    Returns tuple (is_valid: bool, status_code: int, verdict: str)
    where verdict describes why the link is valid/invalid.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}
    # Try once, then a sanitized retry to drop stray trailing punctuation
    tried_urls = [url, url.rstrip(').,;!:"\'')]
    last_exc = None
    r = None

    for u in tried_urls:
        try:
            r = requests.get(u, timeout=timeout, allow_redirects=True, headers=headers)
            break
        except requests.RequestException as e:
            last_exc = e
            r = None

    if r is None:
        # Network problem: be strict for hallucination detection
        print(f"    Request error checking {url}: {last_exc}")
        return False, 0, f"Network error: {last_exc}"

    print(f"    Status code: {r.status_code}")
    # Log a small prefix of content to keep logs readable
    preview = (r.text[:500] + '...') if len(r.text) > 500 else r.text
    # print(f"    Content: {preview}")

    # Hard errors
    if r.status_code in (404, 410):
        return False, r.status_code, "Page not found"
    if r.status_code >= 500:
        return False, r.status_code, "Server error"

    # Bot walls / throttling / auth required
    if r.status_code in (401, 403, 429) or is_bot_challenge(r.text, r.headers, url):
        doi = extract_doi_from_url(url)
        if doi:
            doi_valid = resolve_doi_exists(doi, timeout=timeout)
            verdict = f"Bot wall/Auth required - DOI {'valid' if doi_valid else 'invalid'}"
            return doi_valid, r.status_code, verdict
        return False, r.status_code, "Bot wall/Auth required - no DOI"

    # Success codes: ensure it's not an error shell
    if r.status_code == 200:
        if check_content_for_not_found(r.text, url):
            return False, r.status_code, "Success status but error page content"
        return True, r.status_code, "Valid page"
    
    # Other status codes
    if check_content_for_not_found(r.text, url):
        return False, r.status_code, "Error page content"

    return True, r.status_code, "Accessible page"


def check_folder_already_processed(folder_name, link_log_path, min_entries=5):
    """
    Check if a folder already has at least min_entries in the link log.
    Returns True if folder should be skipped (already processed).
    """
    if not os.path.exists(link_log_path):
        return False
    
    try:
        with open(link_log_path, 'r', encoding='utf-8') as f:
            count = 0
            for line in f:
                # Skip header lines
                if line.startswith('Timestamp') or line.startswith('='):
                    continue
                # Check if line contains our folder name
                parts = line.split(' | ')
                if len(parts) >= 2 and parts[1].strip() == folder_name:
                    count += 1
                    if count >= min_entries:
                        return True
        return False
    except Exception as e:
        print(f"  Warning: Could not read existing link log {link_log_path}: {e}")
        return False


def process_folder(folder_path, delay=1, timeout=10, link_logger=None):
    """Process a single folder containing combined.txt and return statistics."""
    combined_file = folder_path / "combined.txt"

    if not combined_file.exists():
        print(f"  Warning: {combined_file} does not exist")
        return None

    # Read the file content
    try:
        with open(combined_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"  Error reading {combined_file}: {e}")
        return None

    # Extract links
    links = extract_links(content)
    print(f"  Found {len(links)} unique links")

    if not links:
        return {
            'folder_name': folder_path.name,
            'total_links': 0,
            'valid_links': 0,
            'invalid_links': 0,
            'fake_percentage': 0.0
        }

    # Check each link
    valid_count = 0
    invalid_count = 0

    for i, link in enumerate(links, 1):
        print(f"    Checking link {i}/{len(links)}: {link[:80]}{'...' if len(link) > 80 else ''}")

        is_valid, status_code, verdict = check_link_validity(link, timeout=timeout)
        
        # Log to file in real-time
        if link_logger:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            validity = "valid" if is_valid else "invalid"
            link_logger.info(f"{timestamp} | {folder_path.name} | {link} | {status_code} | {validity} | {verdict}")
        
        if is_valid:
            valid_count += 1
            print(f"    ✓ Valid")
        else:
            invalid_count += 1
            print(f"    ✗ Invalid")

        # Be respectful to servers
        if i < len(links):
            time.sleep(delay)

    fake_percentage = (invalid_count / len(links)) * 100 if len(links) > 0 else 0.0

    return {
        'folder_name': folder_path.name,
        'total_links': len(links),
        'valid_links': valid_count,
        'invalid_links': invalid_count,
        'fake_percentage': fake_percentage
    }


def main():
    parser = argparse.ArgumentParser(description='Check citation hallucination by validating links in combined.txt files')
    parser.add_argument('--folder', help='Path to folder containing subfolders with combined.txt files')
    parser.add_argument('--output', default='citation_hallucination_log.txt', help='Output log file (default: citation_hallucination_log.txt)')
    parser.add_argument('--link-log', default='link_processing_log.txt', help='Real-time link processing log (default: link_processing_log.txt)')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between requests in seconds (default: 1.0)')
    parser.add_argument('--timeout', type=int, default=10, help='Request timeout in seconds (default: 10)')

    args = parser.parse_args()

    base_folder = Path(args.folder)
    if not base_folder.exists():
        print(f"Error: Folder {base_folder} does not exist")
        sys.exit(1)

    # Setup link processing logger
    link_logger = logging.getLogger('link_processor')
    link_logger.setLevel(logging.INFO)
    
    # Use append mode if file exists, write mode if not
    log_exists = os.path.exists(args.link_log)
    mode = 'a' if log_exists else 'w'
    
    link_handler = logging.FileHandler(args.link_log, mode=mode, encoding='utf-8')
    link_handler.setLevel(logging.INFO)
    link_formatter = logging.Formatter('%(message)s')
    link_handler.setFormatter(link_formatter)
    link_logger.addHandler(link_handler)
    
    # Write header to link log only if file is new
    if not log_exists:
        link_logger.info("Timestamp | Folder | URL | HTTP_Code | Verdict | Details")
        link_logger.info("=" * 100)

    # Get all subdirectories and sort alphabetically
    subdirs = [d for d in base_folder.iterdir() if d.is_dir()]
    subdirs = sorted(subdirs, key=lambda x: x.name)

    if not subdirs:
        print(f"Error: No subdirectories found in {base_folder}")
        sys.exit(1)

    print(f"Found {len(subdirs)} subdirectories to process")
    print(f"Link processing will be logged to: {args.link_log}")

    # Process each folder
    results = []

    for i, subdir in enumerate(subdirs, 1):
        print(f"\nProcessing folder {i}/{len(subdirs)}: {subdir.name}")
        
        # Check if folder already has enough entries in the log
        if check_folder_already_processed(subdir.name, args.link_log, min_entries=5):
            print(f"  Skipping {subdir.name} - already has 5+ entries in link log")
            continue

        result = process_folder(subdir, delay=args.delay, timeout=args.timeout, link_logger=link_logger)
        if result is not None:
            results.append(result)
            print(f"  Result: {result['invalid_links']}/{result['total_links']} invalid links ({result['fake_percentage']:.1f}% fake)")

    if not results:
        print("No results to process")
        sys.exit(1)

    # Calculate overall statistics
    total_links_overall = sum(r['total_links'] for r in results)
    total_invalid_overall = sum(r['invalid_links'] for r in results)
    overall_fake_percentage = (total_invalid_overall / total_links_overall) * 100 if total_links_overall > 0 else 0.0

    # Calculate average fake percentage across folders
    fake_percentages = [r['fake_percentage'] for r in results if r['total_links'] > 0]
    average_fake_percentage = statistics.mean(fake_percentages) if fake_percentages else 0.0

    # Write results to log file
    output_file = Path(args.output)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Citation Hallucination Analysis Results\n")
        f.write("=" * 40 + "\n\n")

        f.write("Per-folder Results:\n")
        f.write("-" * 20 + "\n")
        for result in results:
            f.write(f"{result['folder_name']}: {result['invalid_links']}/{result['total_links']} invalid links "
                    f"({result['fake_percentage']:.1f}% fake)\n")

        f.write(f"\nOverall Statistics:\n")
        f.write("-" * 18 + "\n")
        f.write(f"Total folders processed: {len(results)}\n")
        f.write(f"Total links checked: {total_links_overall}\n")
        f.write(f"Total invalid references: {total_invalid_overall}\n")
        f.write(f"Overall invalid reference rate: {overall_invalid_percentage:.2f}%\n")
        f.write(f"Average invalid reference rate (across folders): {average_invalid_percentage:.2f}%\n")

        if invalid_percentages:
            f.write(f"Median invalid reference rate: {statistics.median(invalid_percentages):.2f}%\n")
            f.write(f"Min invalid reference rate: {min(invalid_percentages):.2f}%\n")
            f.write(f"Max invalid reference rate: {max(invalid_percentages):.2f}%\n")

        f.write(f"\nFolders with references: {len(invalid_percentages)}\n")
        f.write(f"Folders without references: {len(results) - len(invalid_percentages)}\n")

    print(f"\n=== SUMMARY ===")
    print(f"Total folders processed: {len(results)}")
    print(f"Total links checked: {total_links_overall}")
    print(f"Total invalid references: {total_invalid_overall}")
    print(f"Overall invalid reference rate: {overall_invalid_percentage:.2f}%")
    print(f"Average invalid reference rate (across folders): {average_invalid_percentage:.2f}%")
    print(f"Results saved to: {output_file}")
    print(f"Link processing log saved to: {args.link_log}")
    
    # Close the link logger
    for handler in link_logger.handlers:
        handler.close()
        link_logger.removeHandler(handler)


if __name__ == "__main__":
    main()
