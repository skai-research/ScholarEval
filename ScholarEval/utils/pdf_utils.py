import os
import asyncio
import aiohttp
import requests
import re
import subprocess
from typing import List, Optional, Tuple, Dict, Any
from urllib.parse import urljoin
import json

class FastPDFDownloader:
    def __init__(self, max_workers=80, timeout=8, pdf_dir="pdfs", email=None, log_path=None):
        self.max_workers = max_workers
        self.timeout = timeout
        self.pdf_dir = pdf_dir
        self.email = email  # Required for Unpaywall API
        self.unpaywall_base_url = "https://api.unpaywall.org/v2"

    def _normalize_pdf_url(self, pdf_url: str) -> str:
        """Quick URL normalization for common cases"""
        # Handle arxiv URLs
        if "arxiv.org/abs/" in pdf_url:
            return pdf_url.replace("/abs/", "/pdf/") + ".pdf"

        # Handle ACL anthology URLs
        if "aclanthology" in pdf_url and not pdf_url.endswith(".pdf"):
            return pdf_url + ".pdf"

        # Handle PMC URLs
        if "pmc.ncbi.nlm.nih.gov/articles/PMC" in pdf_url:
            pmc_id = re.search(r"PMC(\d+)", pdf_url)
            if pmc_id:
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmc_id.group(1)}/pdf/"
        elif "ncbi.nlm.nih.gov/pmc/articles/PMC" in pdf_url and not pdf_url.endswith(
            "/pdf/"
        ):
            if not pdf_url.endswith("/pdf/"):
                return pdf_url.rstrip("/") + "/pdf/"

        return pdf_url

    def _is_valid_pdf_content(self, content: bytes) -> bool:
        """Quick PDF validation"""
        if not content or len(content) < 4:
            return False
        return content.startswith(b"%PDF") or content.lstrip().startswith(b"%PDF")

    def _extract_doi_from_url(self, url: str) -> Optional[str]:
        """Extract DOI from various URL formats"""
        # Common DOI patterns
        doi_patterns = [
            r"doi\.org/(.+)",
            r"dx\.doi\.org/(.+)",
            r"doi:(.+)",
            r"/doi/(?:abs/|full/)?(.+)",
            r"DOI:?\s*(.+)",
        ]

        for pattern in doi_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                doi = match.group(1).strip()
                # Clean up common suffixes
                doi = re.sub(r"[#\?].*$", "", doi)
                return doi

        return None

    async def _query_unpaywall(
        self, session: aiohttp.ClientSession, doi: str
    ) -> Optional[Dict[str, Any]]:
        """Query Unpaywall API for open access information"""
        if not self.email:
            print("⚠️  Email required for Unpaywall API")
            return None

        try:
            url = f"{self.unpaywall_base_url}/{doi}?email={self.email}"
            async with session.get(url, ssl=False) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                elif response.status == 404:
                    # DOI not found in Unpaywall
                    return None
                else:
                    print(f"⚠️  Unpaywall API error {response.status} for DOI: {doi}")
                    return None
        except Exception as e:
            print(f"⚠️  Unpaywall query failed for {doi}: {str(e)}")
            return None

    def _get_best_oa_location(self, unpaywall_data: Dict[str, Any]) -> Optional[str]:
        """Extract the best open access PDF URL from Unpaywall response"""
        if not unpaywall_data.get("is_oa", False):
            return None

        oa_locations = unpaywall_data.get("oa_locations", [])
        if not oa_locations:
            return None

        # Prioritize repository sources over publisher sources
        repository_sources = [
            loc for loc in oa_locations if loc.get("host_type") == "repository"
        ]
        publisher_sources = [
            loc for loc in oa_locations if loc.get("host_type") == "publisher"
        ]

        # Try repository sources first (usually more reliable)
        for location in repository_sources + publisher_sources:
            pdf_url = location.get("url_for_pdf")
            if pdf_url:
                return pdf_url

        # Fallback to any URL that might work
        for location in oa_locations:
            url = location.get("url")
            if url and (".pdf" in url.lower() or "pdf" in url.lower()):
                return url

        return None

    async def _try_unpaywall_download(
        self, session: aiohttp.ClientSession, pdf_url: str, pdf_fp: str, corpus_id: str
    ) -> Optional[str]:
        """Try to download PDF using Unpaywall open access information"""

        # Extract DOI from the URL
        doi = self._extract_doi_from_url(pdf_url)
        if not doi:
            return None

        print(f"🔍 Checking Unpaywall for DOI: {doi}")

        # Query Unpaywall API
        unpaywall_data = await self._query_unpaywall(session, doi)
        if not unpaywall_data:
            return None

        # Get the best open access PDF URL
        oa_pdf_url = self._get_best_oa_location(unpaywall_data)
        if not oa_pdf_url:
            print(f"📝 No open access PDF found for: {doi}")
            return None

        print(f"🎯 Found open access PDF: {oa_pdf_url}")

        # Try to download the open access PDF
        result = await self._try_aiohttp_strategies(
            session, oa_pdf_url, pdf_fp, corpus_id
        )
        if result:
            print(f"✅ Unpaywall success: {corpus_id}")
            return result

        # Fallback to requests if aiohttp fails
        result = await self._requests_fallback(oa_pdf_url, pdf_fp, corpus_id)
        if result:
            print(f"✅ Unpaywall + requests: {corpus_id}")
            return result

        return None

    async def _try_aiohttp_strategies(
        self, session: aiohttp.ClientSession, pdf_url: str, pdf_fp: str, corpus_id: str
    ) -> Optional[str]:
        """Try different aiohttp approaches quickly"""

        strategies = [
            # Strategy 1: Basic download
            lambda: session.get(pdf_url, ssl=False),
            # Strategy 2: With browser headers
            lambda: session.get(
                pdf_url,
                ssl=False,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Referer": "https://www.google.com/",
                },
            ),
            # Strategy 3: Allow more redirects
            lambda: session.get(
                pdf_url,
                ssl=False,
                allow_redirects=True,
                max_redirects=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; PDFBot/1.0)"},
            ),
        ]

        for i, strategy in enumerate(strategies):
            try:
                async with strategy() as response:
                    if response.status == 200:
                        content = await response.read()
                        if self._is_valid_pdf_content(content):
                            with open(pdf_fp, "wb") as f:
                                f.write(content)
                            return pdf_fp
                    elif response.status == 429:
                        await asyncio.sleep(1)  # Brief pause for rate limiting
            except Exception:
                continue

        return None

    async def _requests_fallback(
        self, pdf_url: str, pdf_fp: str, corpus_id: str
    ) -> Optional[str]:
        """Fast requests fallback for stubborn URLs"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://google.com",
            }

            # Special handling for known difficult sites
            if "dl.acm.org" in pdf_url:
                headers["Referer"] = "https://dl.acm.org/"
            elif "mdpi.com" in pdf_url:
                headers["Referer"] = "https://www.mdpi.com/"

            # Run in thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: requests.get(
                    pdf_url,
                    headers=headers,
                    timeout=self.timeout,
                    allow_redirects=True,
                    verify=False,
                ),
            )

            if response.status_code == 200:
                content = response.content
                if self._is_valid_pdf_content(content):
                    with open(pdf_fp, "wb") as f:
                        f.write(content)
                    return pdf_fp

        except Exception:
            pass

        return None

    async def _handle_doi_fast(
        self, session: aiohttp.ClientSession, doi_url: str, pdf_fp: str, corpus_id: str
    ) -> Optional[str]:
        """Fast DOI handling with common patterns"""
        try:
            # Follow DOI redirect quickly
            async with session.get(
                doi_url, ssl=False, allow_redirects=True
            ) as response:
                if response.status != 200:
                    return None

                final_url = str(response.url)

                # Try common PDF URL patterns
                pdf_candidates = []

                if "nature.com" in final_url:
                    pdf_candidates.append(final_url.rstrip("/") + ".pdf")
                elif "mdpi.com" in final_url:
                    pdf_candidates.append(final_url.rstrip("/") + "/pdf")
                elif "springer.com" in final_url:
                    pdf_candidates.append(
                        final_url.replace("/article/", "/content/pdf/") + ".pdf"
                    )

                # Try the resolved URL itself
                pdf_candidates.append(final_url)

                # Quick attempt at each candidate
                for candidate_url in pdf_candidates[
                    :2
                ]:  # Limit to 2 attempts for speed
                    try:
                        async with session.get(
                            candidate_url, ssl=False
                        ) as pdf_response:
                            if pdf_response.status == 200:
                                content = await pdf_response.read()
                                if self._is_valid_pdf_content(content):
                                    with open(pdf_fp, "wb") as f:
                                        f.write(content)
                                    return pdf_fp
                    except Exception:
                        continue

        except Exception:
            pass

        return None

    async def download_pdf_async(
        self,
        session: aiohttp.ClientSession,
        pdf_url: str,
        corpus_id: str,
        save_dir: Optional[str] = None,
        try_unpaywall: bool = True,
    ) -> Optional[str]:
        """Download with essential fallbacks and Unpaywall integration"""

        # Use instance pdf_dir if save_dir not provided
        if save_dir is None:
            save_dir = self.pdf_dir

        # Normalize URL
        pdf_url = self._normalize_pdf_url(pdf_url)

        os.makedirs(save_dir, exist_ok=True)
        pdf_fp = os.path.join(save_dir, f"{corpus_id}.pdf")

        # Skip if already exists
        if os.path.exists(pdf_fp):
            return pdf_fp

        # Try Unpaywall first if enabled and URL contains a DOI
        if (
            try_unpaywall
            and self.email
            and ("doi.org" in pdf_url or self._extract_doi_from_url(pdf_url))
        ):
            result = await self._try_unpaywall_download(
                session, pdf_url, pdf_fp, corpus_id
            )
            if result:
                return result

        # Handle DOI URLs with fast resolution
        if "doi.org/" in pdf_url or pdf_url.startswith("10."):
            if not pdf_url.startswith("http"):
                pdf_url = f"https://doi.org/{pdf_url}"
            result = await self._handle_doi_fast(session, pdf_url, pdf_fp, corpus_id)
            if result:
                print(f"✓ DOI: {corpus_id}")
                return result

        # Try aiohttp strategies
        result = await self._try_aiohttp_strategies(session, pdf_url, pdf_fp, corpus_id)
        if result:
            print(f"✓ aiohttp: {corpus_id}")
            return result

        # Requests fallback for stubborn URLs
        result = await self._requests_fallback(pdf_url, pdf_fp, corpus_id)
        if result:
            print(f"✓ requests: {corpus_id}")
            return result

        print(f"✗ Failed: {corpus_id}")
        return None

    async def download_pdfs_batch_async(
        self,
        pdf_data: List[Tuple[str, str]],
        save_dir: Optional[str] = None,
        try_unpaywall: bool = True,
    ) -> List[Optional[str]]:
        """Fast batch download with controlled concurrency and Unpaywall support"""

        # Use instance pdf_dir if save_dir not provided
        if save_dir is None:
            save_dir = self.pdf_dir

        # Balanced connector settings - fast but not overwhelming
        connector = aiohttp.TCPConnector(
            limit=self.max_workers,
            limit_per_host=8,  # Conservative per host to avoid blocks
            ssl=False,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=30,
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout, connect=4)

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf,*/*;q=0.8",
        }

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout, headers=headers
        ) as session:

            # Use semaphore to prevent overwhelming servers
            semaphore = asyncio.Semaphore(self.max_workers)

            async def download_with_semaphore(pdf_url, corpus_id):
                async with semaphore:
                    return await self.download_pdf_async(
                        session, pdf_url, corpus_id, save_dir, try_unpaywall
                    )

            tasks = [
                download_with_semaphore(pdf_url, corpus_id)
                for pdf_url, corpus_id in pdf_data
            ]

            unpaywall_status = (
                "with Unpaywall"
                if try_unpaywall and self.email
                else "without Unpaywall"
            )
            print(f"🚀 Starting {len(tasks)} downloads {unpaywall_status}...")

            # Process in reasonable batches to avoid timeouts
            batch_size = 50
            results = []

            for i in range(0, len(tasks), batch_size):
                batch = tasks[i : i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                results.extend(batch_results)

                # Brief pause between large batches
                if len(batch) == batch_size and i + batch_size < len(tasks):
                    await asyncio.sleep(0.5)

            # Count successes
            successful = sum(1 for r in results if r and not isinstance(r, Exception))
            print(f"✅ Completed: {successful}/{len(tasks)} successful downloads")

            return results

    # Convenience function
    async def fast_download_pdfs(
        pdf_data: List[Tuple[str, str]],
        save_dir: str = "pdfs",
        max_workers: int = 80,
        timeout: int = 8,
        email: Optional[str] = None,
        try_unpaywall: bool = True,
    ) -> List[Optional[str]]:
        """
        Download PDFs with good balance of speed and success rate

        Args:
            pdf_data: List of (pdf_url, corpus_id) tuples
            save_dir: Directory to save PDFs
            max_workers: Max concurrent downloads (default: 80)
            timeout: Timeout per request in seconds (default: 8)
            email: Email for Unpaywall API (required for Unpaywall functionality)
            try_unpaywall: Whether to try Unpaywall for open access versions

        Returns:
            List of file paths (None for failed downloads)
        """
        downloader = FastPDFDownloader(
            max_workers=max_workers, timeout=timeout, pdf_dir=save_dir, email=email
        )
        return await downloader.download_pdfs_batch_async(
            pdf_data, save_dir, try_unpaywall
        )

    def is_grobid_container_running(self):
        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "--filter",
                    "ancestor=lfoppiano/grobid:latest-crf",
                    "--format",
                    "{{.ID}}",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def extract_url(self, text):
        """Extract URL from a given text string using regex."""
        url_pattern = r"https?://[^\s,]+"
        match = re.findall(url_pattern, text)
        if match:
            return match[-1]
        return None

    # Additional utility methods for Unpaywall integration
    async def check_open_access_status(self, doi: str) -> Optional[Dict[str, Any]]:
        """Check if a DOI has open access versions available via Unpaywall"""
        if not self.email:
            print("⚠️  Email required for Unpaywall API")
            return None

        connector = aiohttp.TCPConnector(ssl=False)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(
            connector=connector, timeout=timeout
        ) as session:
            return await self._query_unpaywall(session, doi)

    def get_unpaywall_status_summary(
        self, unpaywall_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get a summary of the open access status from Unpaywall data"""
        if not unpaywall_data:
            return {"is_oa": False, "oa_type": None, "sources": []}

        is_oa = unpaywall_data.get("is_oa", False)
        oa_type = (
            unpaywall_data.get("best_oa_location", {}).get("license") if is_oa else None
        )

        sources = []
        for location in unpaywall_data.get("oa_locations", []):
            sources.append(
                {
                    "host_type": location.get("host_type"),
                    "url": location.get("url"),
                    "pdf_url": location.get("url_for_pdf"),
                    "license": location.get("license"),
                }
            )

        return {
            "is_oa": is_oa,
            "oa_type": oa_type,
            "sources": sources,
            "doi": unpaywall_data.get("doi"),
        }