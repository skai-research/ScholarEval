import json
import argparse
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils
import os

def extract_dimensions_contributions(llm: LLMEngine, research_plan: str) -> tuple:
    """
    Prompts the LLM to extract only the core contributions from the research plan.
    """
    prompt = [
        {"role": "system", "content": (
            "You are a helpful assistant that reads scientific research plans and extracts a structured summary of their contributions. "
            "Your task is to identify a small number of high-level *contribution dimensions*, and for each dimension, extract one or more specific *contribution statements* that are faithful to the research plan.\n\n"

            "*Contribution dimensions* should represent general categories of scientific contribution that are meaningful and comparable across research plans, regardless of the field. "
            "These might include:\n"
            "- \"methodology\" (e.g., proposing a new method, model, or procedure)\n"
            "- \"application\" (e.g., applying existing methods to a new problem or domain)\n"
            "- \"theoretical contribution\" (e.g., proving a new result, deriving a new model)\n"
            "- \"data\" (e.g., constructing a new dataset, conducting original measurements or surveys)\n"
            "- \"evaluation\" (e.g., designing an experimental protocol, benchmarking a technique)\n"
            "- \"tool or system design\" (e.g., building software, devices, or infrastructure to support research)\n"
            "- \"conceptual framework\" (e.g., introducing a new taxonomy or way of thinking about a problem)\n\n"

            "Do not limit your output to the examples above but rather generate suitable dimensions for the research plan given to you. Only include dimensions that are *actually reflected* in the research plan — do not add generic or speculative categories. Please do not generate redundant dimensions. If two dimensions share very similar or identical contribution statements, merge them into a single dimension.\n\n"
            "Ensure that dimensions match the *type of research being proposed*, and avoid vague or overly broad labels such as \"novelty\" or \"contribution\".\n\n"

            "For each dimension, write one or more *contribution statements* that clearly explain what the research is proposing. "
            "These statements should be precise, self-contained, and informative - make sure they include enough context as they will be used as the basis to generate search queries later on."
            "They should accurately reflect what the research plan says, without paraphrasing too generically or introducing new claims.\n\n"

            "Here are a few illustrative examples:\n"
            "- Dimension: \"methodology\"\n"
            "  Statement: \"Developing a new statistical model to estimate ecological diversity from incomplete sampling.\"\n"
            "- Dimension: \"data\"\n"
            "  Statement: \"Collecting a multi-site dataset of interviews with community health workers.\"\n"
            "- Dimension: \"evaluation\"\n"
            "  Statement: \"Designing a randomized controlled trial to assess the policy's effectiveness.\"\n"
            "- Dimension: \"conceptual framework\"\n"
            "  Statement: \"Proposing a new typology for classifying mechanisms of social influence.\"\n\n"

            "Each dimension may include multiple contribution statements, but do not repeat the same idea across dimensions. "
            "Avoid redundancy, and keep the summary compact and informative.\n\n"
        )},
        {"role": "user", "content": (
            f"Here is the research plan:\n\n{research_plan}\n\n"
            "Return the dimensions and contributions as a JSON object where each key is a dimension name and each value is a list of contribution statements.\n\n"
            "Please output a parseable JSON block as follows:\n"
            "```json\n"
            "{{\n"
            "  \"<dimension_name_1>\": [\n"
            "    \"<contribution_statement_1>\",\n"
            "    \"<contribution_statement_2>\"\n"
            "  ],\n"
            "  \"<dimension_name_2>\": [\n"
            "    \"<contribution_statement_3>\"\n"
            "  ]\n"
            "}}\n"
            "```"
        )}
    ]
    
    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.1)
    return response, input_tokens, output_tokens


def main():
    parser = argparse.ArgumentParser(description="Extract core contributions from a research plan.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the research plan text file.")
    parser.add_argument("--llm_engine_name", type=str, required=True, help="Name of the LLM engine to use (e.g., 'gpt-4o').")
    parser.add_argument("--litellm_name", type=str, required=True, help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514').")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the extracted contributions.")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()

    API_KEY = os.environ.get("API_KEY_1")  
    API_ENDPOINT = os.environ.get("API_ENDPOINT") 
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    su = StringUtils()
    
    # Get LLM cost information
    llm_cost = model_cost[args.litellm_name]

    # Read research plan
    with open(args.input_file, "r", encoding="utf-8") as f:
        research_plan = f.read()

    # Extract contributions
    contributions_text, input_tokens, output_tokens = extract_dimensions_contributions(llm, research_plan)
    
    # Calculate cost
    if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
        cost = 0
    else:
        cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                llm_cost["output_cost_per_token"] * output_tokens)
    
    print(f"LLM cost for extract_dimensions_and_contributions: ${cost:.6f}")
    
    # Log cost to centralized cost log if provided
    if args.cost_log_file:
        cost_entry = {
            "step": "extract_dimensions_and_contributions",
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')
    
    # Extract JSON from the response
    contributions_json = su.extract_json_output(contributions_text)
    
    with open(args.output_file, "w", encoding="utf-8") as f:
        for dimension, contributions in contributions_json.items():
            item = {
                "dimension": dimension, 
                "contributions": contributions
            }
            f.write(json.dumps(item) + "\n")

    print(f"Dimensions and contributions saved to: {args.output_file}")


if __name__ == "__main__":
    main()