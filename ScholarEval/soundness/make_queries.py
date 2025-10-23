# steps/make_queries.py
import os, json
from tqdm import tqdm
import argparse
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils

def main():
    parser = argparse.ArgumentParser(description="Extract core contributions from a research plan.")
    parser.add_argument("--research_plan", required=True, help="Path to research plan file")
    parser.add_argument("--methods_file", required=True, help="Path to methods JSON file")
    parser.add_argument("--output_file", required=True, help="Path to output queries JSON")
    parser.add_argument("--llm_engine_name", required=True, help="llm engine name (e.g., 'gpt-4o')")
    parser.add_argument("--litellm_name", required=True, help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()
    API_KEY = os.environ.get("API_KEY")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    su = StringUtils()
    
    llm_cost = model_cost[args.litellm_name]
    
    with open(args.research_plan, "r", encoding="utf-8") as f:
        rp = f.read()

    with open(args.methods_file) as f:
        clean_methods = json.load(f)['clean_methods']

    queries = {}
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for method in tqdm(clean_methods, desc='Writing Queries'):        
        prompt = [
        {
            "role" : "user", 
            "content" : """You are an expert research assistant. Given a method (i.e. one approach that researchers are adopting to execute their idea) extracted from a research plan, please construct a singular query that will be used to search for paper snippets using the semantic scholar API. Use JSON format with 70 words or less per query. Do not include any text in the query besides the query itself. Do not include text like "semantic search query about ..." or "papers related to ...". Just the actual query text. No operators such as AND, OR should be used. Just a query in natural language that is relevant to the method.

            [start extracted method]
            {clean_method}
            [end extracted method]

            In case the method does not have enough context to construct an effective snippet search query, you can use the research plan to understand the overall research direction and inject useful context. 
            [start research plan]
            {rp}                                    
            [end research plan]

            Please output a parseable JSON block as follows, being especially careful to use the correct number of escape characters:
            ```json
            {{
                "query": "Your search query here (IN 100 WORDS OR LESS)"
            }}
            ```""".format(
                clean_method=method,
                rp = rp
            )
        }
        ]
        response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.3)
        
        if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
            method_cost = 0
        else:
            method_cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                          llm_cost["output_cost_per_token"] * output_tokens)
        total_cost += method_cost
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        clean_query = su.extract_json_output(response)['query']
        queries[method] = clean_query

    print(f"Make queries cost: ${total_cost:.4f} (Input: {total_input_tokens}, Output: {total_output_tokens})")

    if args.cost_log_file:
        cost_entry = {
            "step": "make_queries",
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')

    with open(args.output_file, 'w') as f:
        json.dump({'queries': queries}, f, indent=4)

if __name__ == '__main__':
    main()
