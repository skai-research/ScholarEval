import openai
import os
import json
import argparse
import numpy as np
import threading
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity

# Thread-local storage for OpenAI clients
thread_local = threading.local()

def get_thread_client():
    """Get thread-local OpenAI client"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = openai.OpenAI(
            api_key=os.environ.get("API_KEY_1"),
            base_url=os.environ.get("API_ENDPOINT")
        )
    return thread_local.client

def get_embedding(text: str) -> List[float]:
    """Get embedding for text using Titan Text Embeddings V2"""
    client = get_thread_client()
    response = client.embeddings.create(
        input=text,
        model="Titan Text Embeddings V2",
        encoding_format=None 
    )
    return response.data[0].embedding

def load_research_plan(file_path: str) -> str:
    """Load research plan from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def load_paper_metadata(file_path: str) -> List[Dict[str, Any]]:
    """Load paper metadata from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_similarities(plan_embedding: List[float], paper_embeddings: List[List[float]]) -> List[float]:
    """Calculate cosine similarities between plan and papers"""
    plan_array = np.array(plan_embedding).reshape(1, -1)
    papers_array = np.array(paper_embeddings)
    similarities = cosine_similarity(plan_array, papers_array)[0]
    return similarities.tolist()

def filter_top_k_papers(papers: List[Dict[str, Any]], similarities: List[float], k: int) -> List[Dict[str, Any]]:
    """Filter top k papers based on similarity scores"""
    # Create pairs of (similarity, paper) and sort by similarity descending
    paper_sim_pairs = list(zip(similarities, papers))
    paper_sim_pairs.sort(key=lambda x: x[0], reverse=True)
    
    # Add similarity score to each paper and return top k
    top_papers = []
    for sim, paper in paper_sim_pairs[:k]:
        paper_with_score = paper.copy()
        paper_with_score['cosine_similarity'] = sim
        top_papers.append(paper_with_score)
    
    return top_papers

def save_filtered_papers(papers: List[Dict[str, Any]], output_path: str):
    """Save filtered papers to output file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

def main():
    parser = argparse.ArgumentParser(description='Filter papers based on semantic similarity to research plan')
    parser.add_argument('--research_plan', required=True, help='Path to research plan file')
    parser.add_argument('--papers_json', required=True, help='Path to paper metadata JSON file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    parser.add_argument('--top_k', type=int, required=True, help='Number of top papers to keep')
    parser.add_argument('--max_workers', type=int, default=10, help='Maximum number of parallel workers')
    
    args = parser.parse_args()
    
    print(f"Loading research plan from {args.research_plan}...")
    research_plan = load_research_plan(args.research_plan)
    
    print(f"Loading paper metadata from {args.papers_json}...")
    papers = load_paper_metadata(args.papers_json)
    
    # Separate papers with and without relevance_score
    papers_with_score = [paper for paper in papers if 'relevance_score' in paper]
    papers_to_filter = [paper for paper in papers if 'relevance_score' not in paper]
    
    print(f"Found {len(papers_with_score)} papers with existing relevance_score")
    print(f"Found {len(papers_to_filter)} papers to filter using cosine similarity")
    
    if len(papers_to_filter) == 0:
        print("No papers need filtering. Saving papers with existing scores...")
        save_filtered_papers(papers_with_score, args.output)
        print("Done!")
        return
    
    print(f"Generating embedding for research plan...")
    plan_embedding = get_embedding(research_plan)
    
    def get_paper_embedding(paper_info):
        """Get embedding for a single paper"""
        i, paper = paper_info
        abstract = paper.get('abstract', '')
        if not abstract:
            print(f"Warning: Paper {i} has no abstract, using title instead")
            abstract = paper.get('title', '')
        
        embedding = get_embedding(abstract)
        return i, embedding
    
    print(f"Generating embeddings for {len(papers_to_filter)} papers using {args.max_workers} workers...")
    
    # Process only papers that need filtering in parallel
    paper_embeddings = [None] * len(papers_to_filter)  # Pre-allocate to maintain order
    completed_count = 0
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(get_paper_embedding, (i, paper)): i 
            for i, paper in enumerate(papers_to_filter)
        }
        
        # Collect results
        for future in as_completed(future_to_index):
            try:
                i, embedding = future.result()
                paper_embeddings[i] = embedding
                completed_count += 1
                
                if completed_count % 10 == 0 or completed_count == len(papers_to_filter):
                    print(f"Processed {completed_count}/{len(papers_to_filter)} papers")
                    
            except Exception as e:
                i = future_to_index[future]
                print(f"Error processing paper {i}: {e}")
                paper_embeddings[i] = [0.0] * len(plan_embedding)
    
    print("Calculating cosine similarities...")
    similarities = calculate_similarities(plan_embedding, paper_embeddings)
    
    print(f"Filtering top {args.top_k} papers from those without relevance_score...")
    top_filtered_papers = filter_top_k_papers(papers_to_filter, similarities, args.top_k)
    
    # Combine papers with existing scores and newly filtered papers
    all_papers = papers_with_score + top_filtered_papers
    
    print(f"Saving {len(all_papers)} papers to {args.output}...")
    print(f"  - {len(papers_with_score)} papers with existing relevance_score")
    print(f"  - {len(top_filtered_papers)} papers filtered by cosine similarity")
    save_filtered_papers(all_papers, args.output)
    
    print("Done!")
    if top_filtered_papers:
        print(f"Top {len(top_filtered_papers)} filtered papers with similarity scores:")
        for i, paper in enumerate(top_filtered_papers):
            sim = paper['cosine_similarity']
            print(f"{i+1:2d}. {sim:.4f} - {paper['title'][:80]}...")

if __name__ == "__main__":
    main()