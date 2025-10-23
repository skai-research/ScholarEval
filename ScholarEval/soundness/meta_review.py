import os, json
import argparse
import re
from datetime import datetime
from tqdm import tqdm
import time
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils
from ..utils.semantic_scholar import SemanticScholar

def process_reference(rp, method, related_work, rw, llm, su, llm_cost, litellm_name):
    
    prompt = [
        {
            "role": "user", 
            "content": """You are an expert research assistant knowledgeable in many domains. You are extremely critical and observant, and do not overgeneralize findings.
            
            You are given a proposed research method and a list of related work.

            Your objective is to create a meta-review of the related work in the context of the proposed research method. Point out any evidence that supports or contradicts the proposed method. Make sure to contrast the related work as a series of iterative scientific work, where newer work can provide evidence that supports or contradicts older work.

            It is important that the meta-review you generate always ties back to the original research plan. Judge the support and contradictions as well as suggested actions for each method within the general context of the research plan to ensure that your review is highly relevant and precise.

            Be granular, making sure to reference specific details such as:
            - algorithm/technique, datasets/inputs/population, computational resources, statistical methods, implementation details, and evaluation setup, and metrics/instruments used, etc.
            - quantitative outcomes, comparisons to baselines, statistical significance
            - dataset/population size, hardware/system specs, hyperparameters, or other domain-specific constraints that affect reproducibility

            It is important that for each method-level meta review that you generate, you review of the support and contradictions should be ordered starting from strongest evidence of support/contradiction to the weakest. Likewise, the suggested action should be ordered from most important to least important. This does not mean that you will generate these as bullet points, but rather detailed, coherent paragraphs that are logically ordered.

            It is required to copy the in-text citations with their links in markdown format [(author, YYYY-MM)](link) when referring to related work!!

            {related_work}

            [start proposed research method]
            {pm}
            [end proposed research method]

            [start research plan]
            {rp}        
            [end research plan]

            Please output a parseable JSON block as follows (remember it is required to copy the in-text citations with their links in markdown format [(author, YYYY-MM)](link) when referring to related work!!):
            JSON formatting requirements:
            - No trailing commas after the last property
            - Ensure proper closing braces
            - Validate JSON structure before output
            ```json
            {{
                "support": "evidence that supports the proposed method",
                "contradictions": "evidence that contradicts the proposed method",
                "suggested_action": "how can the proposed method be improved based on the related work",
                "soundness_score": "int score 0 to 10 based on the evidence for and against the proposed method"
            }}
            ```
            """.format(related_work=related_work, pm=method, rp=rp)
        }
        ]
    # print(prompt)
    if len(related_work) == 0:
        return {
            'support': 'No related work found',
            'contradictions': 'No related work found',
            'evaluation': 'No related work found',
            'suggested_action': 'No related work found',
            'soundness_score': None,
            'n_related_work': 0,
            'cost': 0,
            'input_tokens': 0,
            'output_tokens': 0
        }
    else:
        print("Processing method:", method)
        response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.3)
        print("Response received", response)
        clean_analysis = su.extract_json_output(response)
        clean_analysis['n_related_work'] = rw
        
        if litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
            cost = 0
        else:
            cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                    llm_cost["output_cost_per_token"] * output_tokens)
        clean_analysis['cost'] = cost
        clean_analysis['input_tokens'] = input_tokens
        clean_analysis['output_tokens'] = output_tokens
        
        return clean_analysis

def convert_json_to_markdown(analysis_data, output_path):
    """Convert the JSON analysis to markdown format"""
    with open(output_path, 'w', encoding='utf-8') as f:
        
        for method_name, analysis in analysis_data.items():
            if (analysis.get('support', '') == 'No related work found' and
                analysis.get('contradictions', '') == 'No related work found' and
                analysis.get('evaluation', '') == 'No related work found' and
                analysis.get('suggested_action', '') == 'No related work found'):
                continue
            else: 
                f.write(f"## {method_name}\n\n")
                
                f.write("### Support\n\n")
                support = analysis.get('support', 'No support information available')
                f.write(f"{support}\n\n")
                
                f.write("### Contradictions\n\n")
                contradictions = analysis.get('contradictions', 'No contradictions information available')
                f.write(f"{contradictions}\n\n")

                f.write("### Suggested Action\n\n")
                suggested_action = analysis.get('suggested_action', 'No suggested action available')
                f.write(f"{suggested_action}\n\n")
                
                f.write("---\n\n")

def save_bibliography(citation_dict, output_path):
    """Save all references used in the meta review to a bibliography text file"""
    s2 = SemanticScholar(os.environ.get("S2_API_KEY_2"))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Bibliography - References Used in Meta Review\n")
        f.write("=" * 50 + "\n\n")
        
        # Sort citations by corpus_id for better organization
        sorted_corpus_ids = []
        for corpus_id, (citation, venue, citation_count) in citation_dict.items():
            # Extract URL from existing citation format [(author, date)](url)
            url_match = re.search(r'\]\(([^)]+)\)', citation)
            url = url_match.group(1) if url_match else "No URL"
            sorted_corpus_ids.append((corpus_id, url))
        
        # Sort by corpus_id
        sorted_corpus_ids.sort(key=lambda x: x[0])
        
        # Write each reference using get_paper_details
        for i, (corpus_id, url) in enumerate(sorted_corpus_ids, 1):
            try:
                # Get paper details using the corpus_id with CorpusId: prefix
                paper_details = s2.get_paper_details(f"CorpusId:{corpus_id}")
                print(f"Paper details fetched: {paper_details}")
                # Extract metadata
                title = paper_details.get('title', 'No Title')
                authors = paper_details.get('authors', [])
                publication_date = paper_details.get('publicationDate', 'No Date')
                venue = paper_details.get('venue', 'No Venue')
                citation_count = paper_details.get('citationCount', 0)
                
                # Format authors
                if authors:
                    author_names = [author.get('name', 'Unknown') for author in authors]
                    if len(author_names) == 1:
                        formatted_authors = author_names[0]
                    elif len(author_names) <= 3:
                        formatted_authors = ", ".join(author_names[:-1]) + f", and {author_names[-1]}"
                    else:
                        formatted_authors = f"{author_names[0]} et al."
                else:
                    formatted_authors = "Unknown Author"
                
                f.write(f"{i}. {formatted_authors} ({publication_date[:4] if publication_date != 'No Date' else 'Unknown Year'})\n")
                f.write(f"   Title: {title}\n")
                f.write(f"   Venue: {venue}\n")
                f.write(f"   Publication Date: {publication_date}\n")
                f.write(f"   Citation Count: {citation_count}\n")
                f.write(f"   URL: {url}\n")
                f.write(f"   Corpus ID: {corpus_id}\n\n")
                time.sleep(1)
            except Exception as e:
                # Fallback to original citation format if API call fails
                print(f"Failed to get details for {corpus_id}: {e}")
                original_citation, original_venue, original_citation_count = citation_dict[corpus_id]
                
                # Extract author and date from original citation
                author_date_match = re.search(r'\[\(([^)]+)\)\]', original_citation)
                if author_date_match:
                    author_date = author_date_match.group(1)
                    f.write(f"{i}. {author_date}\n")
                else:
                    f.write(f"{i}. {original_citation}\n")
                
                f.write(f"   Venue: {original_venue}\n")
                f.write(f"   Citation Count: {original_citation_count}\n")
                f.write(f"   URL: {url}\n")
                f.write(f"   Corpus ID: {corpus_id}\n\n")
        
        f.write(f"\nTotal references: {len(sorted_corpus_ids)}\n")

def main():
    parser = argparse.ArgumentParser(description="Compare methods/goals to snippets")
    parser.add_argument("--research_plan", required=True, help="Path to research plan file")
    parser.add_argument("--mr_analysis_file", required=True, help="Path to methods and results analysis file")
    parser.add_argument("--methods_and_ref_file", required=True, help="Path to methods and references")
    parser.add_argument("--output_file", required=True, help="Path to output methods JSON")
    parser.add_argument("--markdown_output", required=True, help="Path to output markdown file")
    parser.add_argument("--llm_engine_name", required=True, help="llm engine name (e.g., 'gpt-4o')")
    parser.add_argument("--litellm_name", required=True, help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--bibliography_file", required=False, help="Path to save bibliography/references txt file")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()
    
    API_KEY = os.environ.get("API_KEY")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    
    llm_cost = model_cost[args.litellm_name]
    
    with open(args.research_plan, "r", encoding="utf-8") as f:
        rp = f.read()
    with open(args.mr_analysis_file) as f:
        mr_analysis = json.load(f)["analysis"]
    with open(args.methods_and_ref_file) as f:
        clean_methods_refs = json.load(f)
        
    s2 = SemanticScholar(os.environ.get("S2_API_KEY"))  
    
    unique_paper_ids = list(set(["CorpusId:" + ref for references in clean_methods_refs.values() for ref in references]))
    batch_size = 500
    citations = []
    for i in range(0, len(unique_paper_ids), batch_size):
        batch = unique_paper_ids[i:i + batch_size]
        batch_metadata = s2.get_new_citations(batch)
        citations.extend(batch_metadata)
    # remove CorpusId: prefix
    unique_paper_ids = [upid.split(':')[1] for upid in unique_paper_ids]
    
    citation_dict = dict(zip(unique_paper_ids, citations))

    all_methods_analysis = defaultdict(list)
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    cost_lock = threading.Lock()
    
    # Thread-local storage for LLM and StringUtils instances
    thread_local = threading.local()
    
    def get_thread_instances():
        if not hasattr(thread_local, 'llm'):
            thread_local.llm = LLMEngine(
                llm_engine_name=args.llm_engine_name, 
                api_key=API_KEY, 
                api_endpoint=API_ENDPOINT
            )
            thread_local.su = StringUtils()
        return thread_local.llm, thread_local.su
    
    
    def extract_date_from_citation(citation):
        """Extract date from citation format [(author, YYYY-MM)](link)"""
        match = re.search(r'\(.*?,\s*(\d{4}-\d{2})\)', citation)
        if match:
            try:
                return datetime.strptime(match.group(1), '%Y-%m')
            except ValueError:
                return datetime.min  
        return datetime.min  

    def process_ref_wrapper(mr_analysis_tuple):
        method, all_analyses = mr_analysis_tuple
        related_work_data = []
        
        # First pass: collect all valid analyses with their metadata
        for analysis in all_analyses:
            has_data = analysis.get('analysis', None)
            if not has_data:
                continue
            try:
                a = analysis['analysis']
                citation, venue, citation_count = citation_dict[analysis['corpus_id']]
                if isinstance(a, list):
                    good_sample = any([len(item.get(key, '')) > 43 for item in a for key in ["method", "results", "context"]])
                else:
                    good_sample = any([len(a.get(key, '')) > 43 for key in ["method", "results", "context"]])
                if good_sample:
                    related_work_data.append({
                        'citation': citation,
                        'venue': venue,
                        'citation_count': citation_count,
                        'analysis': a if isinstance(a, list) else [a],
                        'date': extract_date_from_citation(citation)
                    })
            except Exception as e:
                print(f"Error processing {analysis}: {e}")
        
        # Sort by release date (oldest first)
        related_work_data.sort(key=lambda x: x['date'])
        
        # Second pass: format the sorted entries
        related_work = []
        for rw, entry in enumerate(related_work_data, 1):
            rw_tmp = f"[start related work {rw}]\n[metadata]\n- in_text_citation: {entry['citation']}\n- venue: {entry['venue']}\n- citation count: {entry['citation_count']}\n"
            for a in entry['analysis']:
                rw_tmp += f"[method]\n{a.get('method', 'no method found')}\n[results]\n{a.get('results', 'no results found')}\n[context]\n{a.get('context', 'no context found')}\n"
            related_work.append(rw_tmp + f"[end related work {rw}]\n")
        related_work = '\n'.join(related_work)
        print(related_work)
        llm, su = get_thread_instances()
        return method, process_reference(rp, method, related_work, len(related_work_data), llm, su, llm_cost, args.litellm_name)
    
        
    # Create tasks for parallel processing
    tasks = [(method, analysis) for method, analysis in mr_analysis.items()]
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_task = {
            executor.submit(process_ref_wrapper, task): task 
            for task in tasks
        }
        
        for future in future_to_task:
            try:
                method_name, result = future.result()
                all_methods_analysis[method_name] = result
                
                with cost_lock:
                    total_cost += result.get('cost', 0)
                    total_input_tokens += result.get('input_tokens', 0)
                    total_output_tokens += result.get('output_tokens', 0)
                    
            except Exception as e:
                task = future_to_task[future]
                print(f"Error processing {task}: {e}")
    
    print(f"Meta review cost: ${total_cost:.4f} (Input: {total_input_tokens}, Output: {total_output_tokens})")
    
    if args.cost_log_file:
        cost_entry = {
            "step": "meta_review",
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')
    
    # Save progress after each method
    with open(args.output_file, 'w') as f:
        json.dump({
            'analysis': all_methods_analysis
        }, f, indent=4)
    
    convert_json_to_markdown(all_methods_analysis, args.markdown_output)
    
    if args.bibliography_file:
        save_bibliography(citation_dict, args.bibliography_file)
    
if __name__ == '__main__':
    main()