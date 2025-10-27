# steps/extract_methods.py
import os, json
import argparse
import pandas as pd
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils

def main():
    parser = argparse.ArgumentParser(description="Compare methods/goals to snippets")
    parser.add_argument("--input_file", required=True, help="Path to input research plan file (parquet)")
    parser.add_argument("--output_file", required=True, help="Path to output methods JSON")
    parser.add_argument("--llm_engine_name", required=True, help="llm engine name (e.g., 'gpt-4o')")
    parser.add_argument("--litellm_name", help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()
    API_KEY = os.environ.get("API_KEY")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    su = StringUtils()
    
    llm_cost = model_cost[args.litellm_name] if args.litellm_name else None
    
    with open(args.input_file, "r", encoding="utf-8") as f:
        rp = f.read()

    prompt = [
        {
            "role" : "user", 
            "content" : """You are an expert research assistant. You are skilled at reading research proposals and identifying the methods that are being proposed to solve the research problem. Methods can be planned system designs, experiments, human studies, analyses, ablations, etc. 
            
            Given a research proposal, you should extract all methods as a python list, such that each method is a separate item in the list. Each item should be a word for word copy of a method, along with a short synthesis that grounds the method in the context of the overall research plan. The extracted methods should be interpretable on their own, this is important.

            Ensure that the methods you extract address different aspects of the research plan and are non-redundant.

            The method list you return should be ranked by importance to the research plan, with the most important methods first. This is crucial. 

            [start research proposal]
            {rp}
            [end research proposal]

            Please output a parseable Python block as follows:
            ```python
            plans = ["context + method", ...]
            ```"""
    .format(
        rp=rp
    )
        }
    ]
    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.1)
    
    if args.litellm_name:
        if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
            cost = 0
        else:
            cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                    llm_cost["output_cost_per_token"] * output_tokens)
        print(f"Extract methods cost: ${cost:.4f} (Input: {input_tokens}, Output: {output_tokens})")
    else:
        cost = 0
        print(f"Extract methods tokens: (Input: {input_tokens}, Output: {output_tokens})")
    
    if args.cost_log_file:
        cost_entry = {
            "step": "extract_methods",
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')
    
    clean_methods = su.extract_python_list(response)
    with open(args.output_file, 'w') as f:
        json.dump({'clean_methods': clean_methods}, f, indent=4)

if __name__ == '__main__':
    main()
