import requests
import os
import json
from typing import List, Dict, Any, Optional, Tuple
import time
import random
import subprocess
import re
import string
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
class SemanticScholar(): 
    """
    A class to interact with the Semantic Scholar API.
    """

    def __init__(self, api_key: str):
        """
        Initializes the SemanticScholar class with the provided API key.

        :param api_key: Your Semantic Scholar API key.
        """
        self.api_key = api_key

    def search_papers(self, query: str):
        """
        Searches for papers using the Semantic Scholar API.

        :param query: The search query.
        :param limit: The maximum number of results to return.
        :return: A list of paper metadata dictionaries.
        """
        # Define the API endpoint URL
        # url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"
        url = "https://api.semanticscholar.org/graph/v1/paper/search"

        # Define the query parameters
        query_params = {
            "query": query,
            "fields": "title,url,abstract,publicationTypes,publicationDate,openAccessPdf",
            "year": "2024-"
        }

        # Define headers with API key
        headers = {"x-api-key": self.api_key}

        # Send the API request
        response = requests.get(url, params=query_params, headers=headers).json()
        return response

    def search_top_papers(
        self,
        query: str,
        limit: int = 20,
        max_date: Tuple[str, None] = None  # e.g. "2023-06-30"
        ) -> List[Dict[str, Any]]:
        """
        Search top 'limit' papers ranked by relevance using Semantic Scholar's search,
        optionally filtering papers published on or before `max_date`.
        """
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "fields": (
                "paperId,title,venue,citationCount,"
                "openAccessPdf,publicationDate,authors,abstract"
            ),
            "limit": limit
        }

        if max_date:
            params["publicationDateOrYear"] = f":{max_date}"

        headers = {"x-api-key": self.api_key}
        time.sleep(1.2)  # Avoid hitting rate limits
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get("data", [])


    def get_recommendations_multi_seed(
        self,
        pos_ids: List[str],
        neg_ids: List[str] = None,
        limit: int = 10,
        max_date: str = None  # format: "YYYY-MM-DD"
        ) -> List[Dict[str, Any]]:
        """
        Get up to 'limit' recommended papers based on positive and optional negative seeds,
        optionally restricted to papers published on or before 'max_date'.
        """
        url = "https://api.semanticscholar.org/recommendations/v1/papers"
        params: Dict[str, Any] = {
            "fields": "paperId,title,venue,citationCount,"
                "openAccessPdf,publicationDate,authors,abstract",
            "limit": limit
        }
        if max_date:
            params["year"] = f":{max_date}"

        headers = {"x-api-key": self.api_key}
        payload: Dict[str, Any] = {"positivePaperIds": pos_ids}
        if neg_ids:
            payload["negativePaperIds"] = neg_ids

        resp = requests.post(url, params=params, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        time.sleep(1)  # Avoid hitting rate limits
        return resp.json().get("recommendedPapers", [])

    def get_influential_references(self, paper_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve influential references for a given paper.
        """
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        params = {
            "fields": "paperId,title,abstract,venue,publicationDate,openAccessPdf,citationCount,influentialCitationCount,isInfluential",
            "limit": 1000
        }
        headers = {"x-api-key": self.api_key}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        refs = response.json().get("data", [])
        return [ref["citedPaper"] for ref in refs if ref.get("isInfluential")]

    def extract_metadata(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and standardize paper metadata.
        """
        return {
            "paperId": paper.get("paperId"),
            "title": paper.get("title"),
            "authors": paper.get("authors"),
            "abstract": paper.get("abstract"),
            "venue": paper.get("venue"),
            "publicationDate": paper.get("publicationDate"),
            "openAccessPdf": paper.get("openAccessPdf", {}),
            "citationCount": paper.get("citationCount"),
        }

    def get_recommendations(self, paper_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Return similar papers using the Recommendations API.
        """
        # paper_id = "34471a2fa18ea22efad5287cf4aeb18542c98a9b"
        url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
        params = {
            # "fields": "paperId,title,abstract,venue,publicationDate,openAccessPdf,citationCount",
            "fields": "paperId,title",
            "limit": limit,
            "from": "all-cs"
        }
        headers = {"x-api-key": self.api_key}
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json().get("data", [])  # list of papers

    def search_snippets_1(self, 
                       query: str, 
                       limit: int = 10,
                       year: Optional[int] = None,
                       fields_of_study: Optional[str] = None,
                       fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Search for text snippets in academic papers.
        
        Args:
            query (str): The search query
            limit (int): Number of results to return (max 100, default 10)
            fields (List[str], optional): Specific fields to return
            
        Returns:
            Dict containing search results with snippets and paper metadata
        """
        url = "https://api.semanticscholar.org/graph/v1/snippet/search"
        headers = {
            'User-Agent': 'SemanticScholarSnippetSearch/1.0',
            'x-api-key': self.api_key
        }
        # Prepare query parameters
        params = {
            'query': query,
            'limit': min(limit, 1000),
        }
        
        if year:
            params['year'] = (
                str(year[0]) + "-" + str(year[1])
                if isinstance(year, (list, tuple)) and len(year) == 2
                else str(year)
            )
        if fields_of_study:
            params['fieldsOfStudy'] = ','.join(fields_of_study)
        if fields:
            params['fields'] = ','.join(fields)
        
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            return {}
    
    def search_snippets(
        self, 
        query: str, 
        limit: int = 10,
        year = None,
        fields_of_study: Optional[str] = None,
        fields: Optional[List[str]] = None,
        rec: bool = False
    ) -> Dict[str, Any]:
        """
        Search for text snippets in academic papers.
        
        Args:
            query (str): The search query
            limit (int): Number of results to return (max 100, default 10)
            fields (List[str], optional): Specific fields to return
            
        Returns:
            Dict containing search results with snippets and paper metadata
        """
        # Prepare query parameters
        print("preparing params")
        params = {
            'query': query,
            'limit': min(limit, 1000),
        }
        print("setting year")
        if year:
            params['publicationDateOrYear'] = (
                year[0] + ":" + year[1]
                if isinstance(year, (list, tuple)) and len(year) == 2 and year[0] and year[1]
                else (":" + str(year[1]) if isinstance(year, (list, tuple)) and len(year) == 2 and not year[0] and year[1] else year)
            )
        if fields_of_study:
            params['fieldsOfStudy'] = ','.join(fields_of_study)
        if fields:
            params['fields'] = ','.join(fields)
        
        time.sleep(1.2)
        url = "https://api.semanticscholar.org/graph/v1/snippet/search"
        headers = {
            'User-Agent': 'SemanticScholarSnippetSearch/1.0',
            'x-api-key': self.api_key
        }
        print(f"Searching snippets with params: {params}")
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=60
        )
        if response.status_code == 200:
            print("Successful response from snippet search")
            return response.json()
        elif response.status_code == 504:
            print(f"\nError: {response.status_code} - {response.text}")
            if rec:
                print('Already tried to retrieve snippets twice, please try again later.')
                return None
            print("Trying again in 10 seconds")
            time.sleep(10)
            return self.search_snippets(query, limit=limit, year=year, fields=fields, rec=True)
        else:
            print(f"\nError: {response.status_code} - {response.text}")
            return None
    
    def get_references(
        self,
        paper_id: str,
        fields: str = "paperId,title,venue,citationCount,openAccessPdf,publicationDate,authors,abstract",
        limit: int = 100,
        max_references: int = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve references of a given paper.

        :param paper_id: Semantic Scholar paper ID
        :param fields: Comma-separated fields to fetch from each citedPaper
        :param limit: number of references per page (≤1000)
        :param max_references: optional cap on total references to fetch
        :return: list of cited paper metadata
        """
        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references"
        headers = {"x-api-key": self.api_key}
        offset = 0
        references: List[Dict[str, Any]] = []

        while True:
            params = {
                "fields": fields,
                "limit": limit,
                "offset": offset
            }
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json().get("data", [])

            if not data:
                break

            for ref in data:
                cited = ref.get("citedPaper")
                if cited:
                    references.append(cited)
                    if max_references and len(references) >= max_references:
                        return references

            offset += len(data)
            # Stop if fewer records than requested were returned (last page)
            if len(data) < limit:
                break
        time.sleep(1)  # Avoid hitting rate limits
        return references
    def get_paper_details(
        self,
        paper_id: str,
        fields: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve detailed metadata about a paper.

        :param paper_id: Semantic Scholar paper ID (or CorpusId, DOI, ArXiv, etc.)
        :param fields: Comma-separated list of fields to return.
                       Defaults to core details: paperId,title,venue,citationCount,
                       openAccessPdf,publicationDate,authors,abstract
        :return: Dict containing paper metadata
        """
        if not fields:
            fields = (
                "paperId,title,venue,citationCount,"
                "openAccessPdf,publicationDate,authors,abstract"
            )

        url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}"
        headers = {"x-api-key": self.api_key}
        params = {"fields": fields}

        resp = requests.get(url, params=params, headers=headers, timeout=30)
        resp.raise_for_status()

        return resp.json()
    def get_single_paper(self, query, fields=None):
        """can search by title"""
        url = f"https://api.semanticscholar.org/graph/v1/paper/search/match"
        if not fields:
            fields = [
                "paperId", "title", "authors", "year", "citationCount", 
                "abstract", "venue", "publicationTypes", "publicationDate"
            ]
        api_key = self.api_key
        if api_key is None:
            raise ValueError("API key must be provided either as an argument or set in the environment variable 's2_key'")
        headers = {
            'User-Agent': 'S2Searcher (OSU)',
            'x-api-key': api_key,
        }
        params = {
            'query': query,
            'limit': 1,
            'fields': ','.join(fields)
        }
        time.sleep(1.2)
        response = requests.get(url, params=params, headers=headers)
        return response.json()
    def is_right_paper(self, abs1, abs2):
        """make sure two abstracts are nearly identical"""
        abs1, abs2 = np.array(abs1.split(' ')), np.array(abs2.split(' '))
        sim_arr = abs1 == abs2
        similarity = np.mean(sim_arr)
        return similarity > 0.95
    def get_safe_pub_date(self, pub_date: str):
        """substract 1 month to make sure we dont accidentally query the paper being evaled"""
        date_obj = datetime.strptime(pub_date, '%Y-%m-%d')
        safe_date = date_obj - relativedelta(months=1)
        return safe_date.strftime('%Y-%m-%d')
    def get_paper_bulk(self, paper_ids, fields=None):
        """Works directly with S2 paper ids"""
        url = "https://api.semanticscholar.org/graph/v1/paper/batch"
        if not fields:
            fields = [
                "paperId", "title", "authors", "year", "citationCount", 
                "abstract", "venue", "publicationTypes", "publicationDate", "url",
                "openAccessPdf", "isOpenAccess"
            ]
        headers = {
            'User-Agent': 'S2Searcher (OSU)',
            'x-api-key': self.api_key,
        }
        params = {
            'fields': ','.join(fields)
        }
        paper_ids_j = {
            "ids": paper_ids
        }
        time.sleep(1.2)
        response = requests.post(url, params=params, headers=headers, json=paper_ids_j)
        return response.json()

    def format_authors(self, authors_list):
        """Format author names for citations"""
        if not authors_list:
            return "unknown_author"
        names = [author['name'].split(' ')[-1] for author in authors_list]

        if not names:
            return "unknown_author"
        elif len(names) == 1:
            return names[0]
        elif len(names) <= 2:
            return ", ".join(names[:-1]) + f", and {names[-1]}"
        else:
            return f"{names[0]} et al."

    def get_new_citations(self, paper_ids):
        """Get formatted citations for papers"""
        info = self.get_paper_bulk(paper_ids, fields=['publicationDate', 'authors', 'url', 'venue', 'citationCount'])
        if len(info) == 0:
            print('ERROR with get_new_citations:', info)
            citations = [''] * len(paper_ids)
            print('S2 rate limit exceeded, filling blank citations')
            return citations
        citations = []
        for paper in info:
            if paper is None or not paper or not info or isinstance(info[0], str):
                paper = {}
                paper['authors'] = [{'name': 'unknown_author'}]
                paper['publicationDate'] = 'unknown_date'
                paper['url'] = '#'
                paper['venue'] = 'No Venue'
                paper['citationCount'] = '0'
            if 'authors' not in paper or not paper['authors']:
                paper['authors'] = [{'name': 'unknown_author'}]
            if 'publicationDate' not in paper or not paper['publicationDate']:
                paper['publicationDate'] = 'unknown_date'
            if 'url' not in paper or not paper['url']:
                paper['url'] = '#'
            if 'venue' not in paper or not paper['venue']:
                paper['venue'] = 'No Venue'
            if 'citationCount' not in paper or not paper['citationCount']:
                paper['citationCount'] = '0'
            citations.append(
                (f"[({self.format_authors(paper['authors'])}, {paper['publicationDate'][:-3]})]({paper['url']})", paper['venue'],paper['citationCount'])
            )
        return citations

    def gen_rand_cit(self) -> List[str]:
        """
        Generate a list of exactly 1000 unique random strings.
        
        Args:
            None
        Returns:
            1000 unique random strings
            
        Raises:
            ValueError: If the character set and length cannot produce 1000 unique combinations
        """
        length = 3
        # Build character set based on parameters
        chars = ""
        
        chars += string.digits  # 0-9
        chars += string.ascii_uppercase  # A-Z
        chars += string.ascii_lowercase  # a-z
        
        # Calculate total possible combinations
        total_combinations = len(chars) ** length
        
        # Check if we can generate 1000 unique strings
        if total_combinations < 1000:
            raise ValueError(f"Cannot generate 1000 unique strings. Only {total_combinations} "
                            f"possible combinations with {len(chars)} characters and length {length}. "
                            f"Try increasing length or expanding character set.")
        
        # Generate unique strings using a set to avoid duplicates
        unique_strings = set()
        
        # Keep generating until we have 1000 unique strings
        while len(unique_strings) < 1000:
            random_string = ''.join(random.choice(chars) for _ in range(length))
            unique_strings.add('<|' + random_string + '|>')
        
        # Convert to list and return
        return list(unique_strings)


    def steal_cite_from_snip(self, snippets_data):
        all_paper_ids = set()
        for i, _ in enumerate(snippets_data):
            # Add current snippet's paper ID
            all_paper_ids.add(snippets_data[i]['paper']['corpusId'])
            # Then add all citations found in the snippet text
            annotations = snippets_data[i]['snippet']['annotations']
            ref_mentions = annotations['refMentions']
            citations = set()
            if ref_mentions:
                for reference in ref_mentions:
                    cite = reference['matchedPaperCorpusId']
                    if cite:
                        citations.add(cite)
            all_paper_ids.update(citations)
        return all_paper_ids

    def match_sentences_with_references(self, annotations, snippet_text, snippet_pid):
        """
        Create tuples matching sentences with their referenced paper IDs.
        For snippet search
        
        Args:
            annotations (dict): Dictionary containing 'refMentions' and 'sentences'
            snippet_text (str, optional): Full text to extract sentences from
            snippet_pid: paper id for snippet being parsed
        
        Returns:
            list: List of tuples (sentence_index, paper_ids_set, cleaned_text)
                  If snippet_text is None, cleaned_text will be None
        """
        ref_mentions = annotations['refMentions']
        sentences = annotations['sentences']
        fake_citations = self.gen_rand_cit()
        snippet_citation = fake_citations[0]
        print(snippet_citation)
        
        if ref_mentions == None or sentences == None:
            return None
        
        # First pass: collect all sentence data and unique paper IDs
        sentence_data = []
        all_paper_ids = set()
        
        for sentence_idx, sentence in enumerate(sentences):
            sentence_start = sentence['start']
            sentence_end = sentence['end']
            
            # Find all references that overlap with this sentence
            paper_ids = []
            sentence_refs = []  # Store refs for this sentence to remove from text
            
            if ref_mentions:
                for ref in ref_mentions:
                    ref_start = ref['start']
                    ref_end = ref['end']
                    paper_id = ref['matchedPaperCorpusId']
                    
                    # Check if reference overlaps with sentence range
                    if (ref_start >= sentence_start and ref_start < sentence_end) or \
                    (ref_end > sentence_start and ref_end <= sentence_end) or \
                    (ref_start <= sentence_start and ref_end >= sentence_end):
                        
                        # Store reference info for text cleaning
                        sentence_refs.append({'start': ref_start, 'end': ref_end})
                        paper_ids.append(paper_id)
                        
                        # Add to set of all paper IDs for batched lookup
                        if paper_id is not None:
                            all_paper_ids.add(paper_id)
            
            # Store sentence data for later processing
            sentence_data.append({
                'sentence_idx': sentence_idx,
                'sentence_start': sentence_start,
                'sentence_end': sentence_end,
                'paper_ids': paper_ids,
                'sentence_refs': sentence_refs
            })
        
        # Batch call to get all citations at once
        all_clean_pids = ["CorpusId:" + str(pid) for pid in all_paper_ids]
        if all_clean_pids:
            # all_citation_info = self.get_new_citations(all_clean_pids)
            all_citation_info = fake_citations[1:len(all_clean_pids)]
            # Create mapping from paper_id to citation
            pid_to_citation = {}
            for i, pid in enumerate(all_paper_ids):
                if i < len(all_citation_info):
                    pid_to_citation[pid] = all_citation_info[i]
                else:
                    pid_to_citation[pid] = ''
        else:
            pid_to_citation = {}
        
        # Second pass: process each sentence with batched citation data
        result = []
        
        for data in sentence_data:
            sentence_idx = data['sentence_idx']
            sentence_start = data['sentence_start']
            sentence_end = data['sentence_end']
            paper_ids = data['paper_ids']
            sentence_refs = data['sentence_refs']
            
            # Get citations for this sentence's paper IDs
            clean_citations = []
            for pid in paper_ids:
                if pid is not None and pid in pid_to_citation:
                    clean_citations.append(pid_to_citation[pid])
                else:
                    clean_citations.append('')
            
            # Extract and clean sentence text if snippet_text provided
            cleaned_text = None
            if snippet_text is not None:
                # Get original sentence text
                original_text = snippet_text[sentence_start:sentence_end]
                
                # Remove citation markers by replacing them with empty string
                cleaned_text = original_text
                
                # Sort references by start position in reverse order to avoid index shifting
                sentence_refs.sort(key=lambda x: x['start'], reverse=True)
                if not sentence_refs:
                    if cleaned_text.strip()[-1] == '.':
                        cleaned_text = cleaned_text.strip()[:-1] + ' ' + snippet_citation + '.'
                    else:
                        cleaned_text = cleaned_text.strip()
                for ref, cit in zip(sentence_refs, clean_citations):
                    # Adjust reference positions relative to sentence start
                    rel_start = ref['start'] - sentence_start
                    rel_end = ref['end'] - sentence_start
                    
                    # Only remove if the reference is within sentence bounds
                    if 0 <= rel_start < len(cleaned_text) and 0 <= rel_end <= len(cleaned_text):
                        if cit == '':
                            cleaned_text = cleaned_text[:rel_start].strip() + ' ' + cleaned_text[rel_end:].strip()
                        elif cleaned_text[rel_end:].strip() == '.':
                            cleaned_text = cleaned_text[:rel_start].strip() + ' ' + cit + cleaned_text[rel_end:].strip()
                        else:
                            cleaned_text = cleaned_text[:rel_start].strip() + ' ' + cit + ' ' + cleaned_text[rel_end:].strip()
            
            # Add tuple of (sentence_index, set_of_paper_ids, cleaned_text)
            result.append((sentence_idx, paper_ids, cleaned_text))
        
        return result

    def format_all_snippets(self, snippets_data, n_snips=0):
        n_top_snippets = len(snippets_data) if n_snips==0 else n_snips
        all_snip = []
        for i in range(n_top_snippets):
            annotations = snippets_data[i]['snippet']['annotations']
            snippet_text = snippets_data[i]['snippet']['text']
            snippet_pid = snippets_data[i]['paper']['corpusId']
            sent_refs = self.match_sentences_with_references(annotations, snippet_text, snippet_pid)
            snip = '[start snippet]'
            if sent_refs:
                for sent_ref in sent_refs:
                    snip += sent_ref[-1] + ' '
                snip = snip.strip() + '[end snippet]'
                all_snip.append(snip)
        return all_snip

    def extract_sections_from_pdf(self, client, pdf_dir):
        import glob
        import os
        
        # Count PDFs to be processed
        pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
        
        # Call GROBID to process PDFs
        try:
            result = client.process(
                "processFulltextDocument",
                pdf_dir,
                n=10  # number of threads
            )
        except Exception as e:
            print(f"GROBID processing failed: {e}")
            result = None
        
        return result


    def is_direct_pdf(self, url):
        """
        Quick check to see if URL points to a downloadable PDF.
        Returns True if it's a direct PDF, False otherwise.
        """
        try:
            # Quick checks first (no network request)
            if 'pdf' in url.lower():
                return True
            
            # HEAD request to check content type (minimal network usage)
            response = requests.head(url, allow_redirects=True, timeout=5)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '').lower()
                if 'application/pdf' in content_type:
                    return True
                
                # Check Content-Disposition for PDF filename
                content_disposition = response.headers.get('content-disposition', '').lower()
                if 'filename' in content_disposition and '.pdf' in content_disposition:
                    return True
            
            return False
        except Exception:
            # If we can't check, assume it's not a direct PDF
            return False
    
    def download_pdf(self, pdf_url, corpus_id, save_dir="pdfs"):
        """Download a PDF from OpenReview using a paper ID and save it to a directory."""
        # This issue appears sometimes with arxiv links
        if '/abs/' in pdf_url:
            pdf_url = pdf_url.replace('/abs/', '/pdf/')
            
        # Quick check - skip if not a direct PDF
        if not self.is_direct_pdf(pdf_url):
            print(f"Skipping {corpus_id} - not a direct PDF URL")
            return None

        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)

        pdf_fp = os.path.join(save_dir, f"{corpus_id}.pdf")
        if os.path.exists(pdf_fp):
            return None
        try:
            response = requests.get(pdf_url)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {corpus_id} -- {pdf_url}: {e}")
            return None
        if response.status_code == 200:
            with open(pdf_fp, "wb") as f:
                f.write(response.content)
            return pdf_fp  # Return file path
        else:
            print(f"Failed to download {corpus_id} -- response.status_code: {response.status_code} -- {pdf_url}")
            return None

    