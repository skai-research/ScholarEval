import os, json
import argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils

def process_reference(rp, method, ref, clean_ref_and_paper, llm, su, llm_cost, litellm_name):
    """Process a single reference for a given method"""
    if ref not in clean_ref_and_paper.keys():
        print('Reference not found in paper text:', ref)
        return {
            'corpus_id': ref,
            'analysis': {
                "method": "N/A",
                "result": "No paper text available for this reference.",
                "context": "N/A"
            },
            'cost': 0,
            'input_tokens': 0,
            'output_tokens': 0
        }
    
    prompt = [
    {
        "role": "user", 
        "content": """You are an expert research assistant knowledgeable in many domains. You are extremely critical and observant, and do not to overgeneralize findings. You are given a proposed research method and the methods/results section from a paper.
        [start paper]
        {paper_text}
        [end paper]

        [start proposed research method]
        {pm}
        [end proposed research method]

        To further understand the scope of the proposed research method, here is the entire research plan that it is extracted from - a method is a single approach that researchers are adopting to execute their research plan:
        [start research plan]
        {rp}        
        [end research plan]

        For any method in the paper that is related to the proposed research method and the overall research plan, please summarize the method used in the paper, report the experimental outcome from using the method, and provide some context for experimental conditions.

        Do not use any in-text citations. Ensure that the method, results, and context you provide are specific and detailed, and has mention of how it relates to the proposed method and research plan.

        If the proposed research method does not relate to any methods in the paper, please return an empty dictionary.

        Strictly follow the output format displayed below.

        JSON formatting requirements:
        - Must be a complete, valid JSON object
        - Start with an open bracket and end with closed bracket
        - No trailing commas after the last property
        - Validate JSON structure before output
        ```json
        {{
            "method": "Description of experimental approach including: algorithm/technique, datasets/inputs, computational resources, implementation/experimentation details, and evaluation setup, and metrics/instruments used, etc.",
            "results": "Quantitative outcomes with specific values, comparisons to baselines, statistical significance where applicable",
            "context": "Key experimental conditions: dataset/population size, hardware/system/instrument specs, hyperparameters, or other domain-specific constraints that affect reproducibility"
        }}
        ```""".format(paper_text=clean_ref_and_paper[ref], pm=method, rp=rp)
                }
    ]

    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.3)
    clean_analysis = su.extract_json_output(response)

    if litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
        cost = 0
    else:
        cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                llm_cost["output_cost_per_token"] * output_tokens)

    return {
        'corpus_id': ref,
        'analysis': clean_analysis,
        'cost': cost,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens
    }

def main():
    parser = argparse.ArgumentParser(description="Compare methods/goals to snippets")
    parser.add_argument("--research_plan", required=True, help="Path to research plan file")
    parser.add_argument("--methods_and_ref_file", required=True, help="Path to methods and references")
    parser.add_argument("--ref_and_paper_file", required=True, help="Path to references and paper text")
    parser.add_argument("--output_file", required=True, help="Path to output methods JSON")
    parser.add_argument("--llm_engine_name", required=True, help="llm engine name (e.g., 'gpt-4o')")
    parser.add_argument("--litellm_name", required=True, help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of parallel workers")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()
    
    API_KEY = os.environ.get("API_KEY")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    
    llm_cost = model_cost[args.litellm_name]
    
    with open(args.research_plan, "r", encoding="utf-8") as f:
        rp = f.read()
    with open(args.methods_and_ref_file) as f:
        clean_methods_refs = json.load(f)
    with open(args.ref_and_paper_file) as f:
        clean_ref_and_paper = json.load(f)

    all_methods_analysis = defaultdict(list)
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    cost_lock = threading.Lock()
    
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
    
    def process_ref_wrapper(method_ref_tuple):
        method, ref = method_ref_tuple
        llm, su = get_thread_instances()
        return method, process_reference(rp, method, ref, clean_ref_and_paper, llm, su, llm_cost, args.litellm_name)
    
    for method, references in tqdm(clean_methods_refs.items(), desc='Analyzing Sources'):
        
        tasks = [(method, ref) for ref in references]
        
        # Process references in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_task = {
                executor.submit(process_ref_wrapper, task): task 
                for task in tasks
            }
            
            # Collect results with progress bar
            for future in future_to_task:
                try:
                    method_name, result = future.result()
                    all_methods_analysis[method_name].append(result)
                    
                    with cost_lock:
                        total_cost += result.get('cost', 0)
                        total_input_tokens += result.get('input_tokens', 0)
                        total_output_tokens += result.get('output_tokens', 0)
                        
                except Exception as e:
                    task = future_to_task[future]
                    print(f"Error processing {task}: {e}")
        
        # Save progress after each method
        with open(args.output_file, 'w') as f:
            json.dump({
                'analysis': all_methods_analysis
            }, f, indent=4)
    
    print(f"Method analysis cost: ${total_cost:.4f} (Input: {total_input_tokens}, Output: {total_output_tokens})")
    
    # Log cost to centralized cost log if provided
    if args.cost_log_file:
        cost_entry = {
            "step": "mr_synth_cond_pm",
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')
        
if __name__ == '__main__':
    main()