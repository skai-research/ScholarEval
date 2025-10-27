import argparse
import json
import os
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine

def generate_queries_for_contribution(llm: LLMEngine, contrib: str, research_plan: str, n_queries: int = 3) -> tuple:
    """
    Use LLM to craft optimized search queries that best capture the contribution in context of the full research plan.
    """
    prompt = [
    {
        "role": "system",
        "content": (
            f"You are an expert at writing highly targeted search queries for retrieving academic papers using the Semantic Scholar API. "
            f"You will be given a full research plan and one specific contribution from that plan. "
            f"Your task is to generate up to {n_queries} short, diverse, and high-quality search queries that are focused on the given contribution and consistent with the overall research context.\n\n"

            "These queries will be used to search for paper abstracts. They must be semantically rich and optimized to retrieve papers that are directly relevant to the contribution. "
            "*Avoid generating queries that are too general, overly broad, or likely to return unrelated results.*\n\n Also avoid generating queries that are just broadly related to the research plan but specifically tailored to the contribution.\n\n"

            "Guidelines:\n"
            "- Each query should be brief and focused (as if typed into a search bar, as a rule of thumb do not exceed 7 words per query).\n"
            "- Queries must stay tightly aligned with the core idea of the contribution.\n"
            "- Incorporate key methods, problems, domains, or goals described in the contribution.\n"
            "- Reflect an understanding of the broader research plan, but do not drift away from the specific contribution.\n"
            "- Use natural phrasing (no Boolean operators like AND, OR, etc.).\n"
            "- Do not include vague terms or general-purpose research language.\n"
            "- Ensure queries are meaningfully different from one another while remaining on-topic.\n\n"

            "Think like a researcher who wants to find papers that specifically address the same problem or idea—not just loosely related topics."
        )
    },
    {
        "role": "user",
        "content": (
            f"Research Plan:\n{research_plan}\n\n"
            f"Contribution:\n{contrib}\n\n"
            "Generate up to {n_queries} precise and optimized search queries:"
            )
        }
    ]

    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.3)
    queries = [line.strip() for line in response.splitlines() if line.strip()]
    return queries[:n_queries], input_tokens, output_tokens

def main():
    parser = argparse.ArgumentParser(description="Generate optimized Semantic Scholar queries from contributions and a research plan.")
    parser.add_argument("--research_plan", type=str, required=True,
                        help="Path to the full research plan text file")
    parser.add_argument("--contrib_file", type=str, required=True,
                        help="Path to .txt file with one contribution per line")
    parser.add_argument("--llm_engine_name", type=str, required=True,
                        help="Name of the LLM engine to use (e.g., 'gpt-4')")
    parser.add_argument("--litellm_name", type=str,
                        help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save generated queries as JSON")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()

    API_KEY = os.environ.get("API_KEY")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    
    # Get LLM cost information
    llm_cost = model_cost[args.litellm_name] if args.litellm_name else None

    with open(args.research_plan, "r", encoding="utf-8") as f:
        research_plan = f.read()

    contributions = []
    with open(args.contrib_file, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            contributions.extend(entry.get("contributions", []))

    all_queries: dict[str, list[str]] = {}
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    
    for contrib in contributions:
        queries, input_tokens, output_tokens = generate_queries_for_contribution(llm, contrib, research_plan)
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
        all_queries[contrib] = queries
    
    print(f"Total LLM cost for contribution_queries_generator: ${total_cost:.6f}")

    # Log cost to centralized cost log if provided
    if args.cost_log_file:
        cost_entry = {
            "step": "contribution_queries_generator",
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')

    # Save output without cost information
    output_data = {
        "queries": all_queries
    }
    
    with open(args.output_file, "w", encoding="utf-8") as outf:
        json.dump(output_data, outf, indent=2, ensure_ascii=False)

    print(f"Generated queries for {len(contributions)} contributions; output saved to {args.output_file}")

if __name__ == "__main__":
    main()
