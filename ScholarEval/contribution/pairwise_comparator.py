import argparse
import json
import logging
import os
import threading
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prompt_pairwise_comparison(llm: LLMEngine, research_plan: str, paper: Dict, aggregated_dimensions:str) -> tuple:
    
    prompt = [
    {
        "role": "system",
        "content": (
            "You are an expert in assessing the NOVELTY of scientific research plans relative to existing work.\n"
            "Your ONLY job is to compare a full research plan to a paper ABSTRACT **purely on novelty and contribution originality**.\n"
            "You must NOT evaluate methodology soundness, feasibility, correctness, or experimental quality—those are outside scope.\n\n"

            "You will be given a research plan, a paper abstract, and a comma-separated list of contribution dimensions.\n\n"

            "Produce a structured output consisting of:\n"
            "1) \"overall_comparison\": a broad yet precise summary of the **novelty** of the research plan versus the paper—i.e., whether the plan proposes ideas/angles/uses that appear original relative to what the abstract claims; identify overlap vs. originality explicitly.\n"
            "2) \"dimension_comparisons\": for EACH provided dimension, a **novelty-only** comparison that states whether the plan is doing something not present in the paper for that dimension (or vice versa), and a numeric score:\n"
            "   • 1  = The research plan is more NOVEL under this dimension (adds ideas/angles/uses not present in the paper abstract)\n"
            "   • 0  = Neither appears more novel OR the paper does not address this dimension (tie/insufficient evidence)\n"
            "   • -1 = The paper appears more NOVEL or the plan largely replicates what the paper already presents under this dimension\n\n"

            "STRICT GUIDELINES (read carefully):\n"
            "- Focus EXCLUSIVELY on novelty/original contribution. DO NOT discuss validity, soundness, experimental design, significance testing, or reliability.\n"
            "- Anchor claims to explicit content in the plan vs. abstract (e.g., unique tasks, datasets/sources, targets, settings, objectives, outputs, frameworks, evaluation angles, applications, constraints, or integration patterns).\n"
            "- Be specific and avoid vague phrasing. If evidence is insufficient, say so and assign 0 for that dimension.\n"
            "- Do not invent details not present in the inputs. If the abstract is silent on a dimension, state this plainly.\n"
            "- Keep writing tight, academic, and to the point.\n\n"

            "Return your output as a parseable JSON block with exactly this structure:\n"
            "```json\n"
            "{{\n"
            "  \"overall_comparison\": \"<novelty-focused summary (no soundness/validity talk)>\",\n"
            "  \"dimension_comparisons\": {\n"
            "    \"<dimension_1>\": {\n"
            "      \"comparison\": \"<novelty-only comparison for this dimension>\",\n"
            "      \"score\": <1 | 0 | -1>\n"
            "    },\n"
            "    \"<dimension_2>\": {\n"
            "      \"comparison\": \"<novelty-only comparison>\",\n"
            "      \"score\": <1 | 0 | -1>\n"
            "    }\n"
            "  }\n"
            "}}\n"
            "```"
        )
    },
    {
        "role": "user",
        "content": (
            f"Here is the research plan:\n\n{research_plan}\n\n"
            f"Here is the paper:\n\n"
            f"Title: {paper.get('title')}\n\n"
            f"Abstract: {paper.get('abstract')}\n\n"
            f"Here are the dimensions to consider (novelty-only for each): {aggregated_dimensions}.\n\n"
            "Please return your output in the specified JSON format."
        )
    }
    ]

    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.3)
    return response.strip(), input_tokens, output_tokens

def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Compare research plan with papers and analyze contribution overlap.")
    parser.add_argument("--research_plan", required=True, help="Path to research plan .txt file.")
    parser.add_argument("--papers_metadata", required=True, help="Path to papers metadata JSON file.")
    parser.add_argument("--dimensions_file", required=True, help="Path to dimensions JSONL file.")
    parser.add_argument("--llm_engine", required=True, help="Name of the LLM engine (e.g., 'gpt-4o').")
    parser.add_argument("--litellm_name", help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514').")
    parser.add_argument("--output_file", required=True, help="Where to save the pairwise comparisons.")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of parallel workers")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()

    logging.info("Loading research plan and papers")
    research_plan_text = load_file(args.research_plan)
    papers = load_json(args.papers_metadata)
    
    # Get LLM cost information
    llm_cost = model_cost[args.litellm_name] if args.litellm_name else None
    
    # Thread-safe cost tracking
    cost_lock = threading.Lock()
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0

    thread_local = threading.local()
    
    def get_thread_instances():
        if not hasattr(thread_local, 'llm'):
            thread_local.llm = LLMEngine(
                llm_engine_name=args.llm_engine,
                api_key=os.environ.get("API_KEY"),
                api_endpoint=os.environ.get("API_ENDPOINT")
            )
            thread_local.su = StringUtils()
        return thread_local.llm, thread_local.su

    valid_papers = [p for p in papers if p.get("abstract")]

    logging.info("Loading dimensions file")
    dimensions = []
    with open(args.dimensions_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                dimensions.append(data.get("dimension", ""))
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse line in dimensions file: {e}")
    
    aggregated_dimensions = ", ".join(dimensions)
    logging.info(f"Aggregated dimensions: {aggregated_dimensions}")

    def compare_paper(paper):
        """Compare a single paper using thread-local LLM instance."""
        nonlocal total_cost, total_input_tokens, total_output_tokens
        logging.info(f"Comparing with paper: {paper.get('title')[:80]}")
        llm, su = get_thread_instances()
        
        try:
            comparison_output, input_tokens, output_tokens = prompt_pairwise_comparison(llm, research_plan_text, paper, aggregated_dimensions)
            
            # Thread-safe cost tracking
            with cost_lock:
                if args.litellm_name:
                    if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
                        cost = 0
                    else:
                        cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                                llm_cost["output_cost_per_token"] * output_tokens)
                else:
                    cost = 0
                total_cost += cost
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
            
            comparison_json = su.extract_json_output(comparison_output)
            return {
                "paper_title": paper.get("title"),
                "paper_authors": paper.get("authors"),
                "paper_abstract": paper.get("abstract"),
                "paper_venue": paper.get("venue"),
                "paper_pdf": paper.get("openAccessPdf"),
                "citation_count": paper.get("citationCount"),
                "publication_date": paper.get("publicationDate"),
                "comparison": comparison_json,
                "cost": cost,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            }
        except Exception as e:
            logging.error(f"Failed comparison for paper '{paper.get('title')}': {e}")
            return {
                "paper_title": paper.get("title"),
                "paper_authors": paper.get("authors"),
                "paper_abstract": paper.get("abstract"),
                "paper_venue": paper.get("venue"),
                "paper_pdf": paper.get("openAccessPdf"),
                "citation_count": paper.get("citationCount"),
                "publication_date": paper.get("publicationDate"),
                "comparison": {"error": f"Comparison failed: {str(e)}"},
                "cost": 0,
                "input_tokens": 0,
                "output_tokens": 0
            }

    # Process papers in parallel
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit all tasks
        future_to_paper = {
            executor.submit(compare_paper, paper): paper 
            for paper in valid_papers
        }
        
        # Collect results
        for future in tqdm(as_completed(future_to_paper), total=len(valid_papers), desc="Comparing papers"):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                paper = future_to_paper[future]
                logging.error(f"Error processing paper {paper.get('title')}: {e}")

    print(f"Total LLM cost for pairwise_comparator: ${total_cost:.6f}")
    logging.info(f"Saving {len(results)} pairwise comparisons to {args.output_file}")
    
    # Log cost to centralized cost log if provided
    if args.cost_log_file:
        cost_entry = {
            "step": "pairwise_comparator",
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')
    
    # Save output without cost summary
    output_data = {
        "comparisons": results
    }
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    logging.info("Done.")

if __name__ == "__main__":
    main()
