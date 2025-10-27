import argparse
import json
import logging
import os
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.citation_check import check_citations_from_strings

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

def load_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def prompt_final_synthesis(llm: LLMEngine, research_plan: str, comparisons: list) -> tuple:
    
    prompt = [
    {
        "role": "system",
        "content": (
            "You are a **professor** reviewing a research proposal **for novelty and originality of its contributions and their impact**. "
            "You are critical, knowledgeable, precise, and not afraid to give truthful assessment of the proposal's novelty. "
            "Your tone is formal, academic, and to the point. "
            "Do **not** discuss methodology soundness, feasibility, correctness, or experimental quality—those are out of scope here.\n\n"

            "You will be given a full research plan and a JSON file of pairwise comparisons between the plan and existing papers. "
            "Each comparison includes a paper reference (authors, title, year, URL), an `overall_comparison` (novelty-focused), "
            "and `dimension_comparisons` comparing the research proposal to a related work based on a specific contribution dimension.\n\n"

            "Your task is to **synthesize an overall evaluation of the proposal's novelty and impact** based on **all** pairwise comparisons.\n\n"

            "STRUCTURE (must adhere exactly):\n"
            "• Produce a novelty and impact assessment for each contribution dimension found in the comparisons. "
            "Each paragraph focuses on one dimension and contains **three subsections**:\n"
            "  - **Strengths**: Precisely explain where the proposal is **novel** under this dimension versus the papers and makes an impact; back claims with in-text citations.\n"
            "  - **Weaknesses**: Precisely explain where the proposal **lacks novelty** or is already covered by prior work, and hence lacks impact; back claims with in-text citations. Be exhaustive with identifying weaknesses.\n\n"
            "  - **Suggestions**: Give actionable, feasible, and useful suggestions to improve the novelty and impact of the work, if needed, based on the evidence from the Strengths and Weaknesses sections; back claims with in-text citations.\n\n"

            "EVIDENCE & REASONING REQUIREMENTS:\n"
            "- Base your analysis on the `overall_comparison` and the `dimension_comparisons`. "
            "- Use your **expert judgment** to interpret these comparisons, but **substantive novelty and impact claims must be citation-backed** via the provided papers. "
            "- Do not merely restate what other works do—**explicitly state how that affects the proposal's novelty and impact**"
            "- Be exhaustive about **novelty and impact weaknesses** that a human evaluator might underscore; avoid vagueness.\n\n"

            "CITATIONS (mandatory):\n"
            "- Use in-text citations **with links** in Markdown format: [Author et al., Year](URL). "
            "- At the end, include a **References** section listing all cited papers using the `paper_reference` fields. "
            "- Only cite only papers present in the comparison data.\n\n"

            "STYLE & SCOPE:\n"
            "- Focus **only** on novelty/originality contribution (what is new vs. already done) and its impact. "
            "- Keep writing formal, objective, and precise. "
            "- To the extent possible given the papers provided to you and your knowledge, give a long, deep, and informative novelty-impact analysis."
            "- Do **not** quote or refer to the pairwise comparisons as a source; synthesize as an independent review informed by them. "
            "- Do **not** output any scores or extra commentary; a separate module handles scoring."
        )
    },
    {
        "role": "user",
        "content": (
            f"Research Plan:\n\n{research_plan}\n\n"
            f"Comparison Data (as JSON):\n\n{json.dumps(comparisons, indent=2)}\n\n"
            "Now synthesize a comprehensive **novelty and impact-focused** evaluation of the research plan's contributions based on the overall and dimension-level comparisons provided. "
            "Remember: one detailed assessment per dimension, each with **Strengths**, **Weaknesses**, and **Suggestions**, precise and citation-backed throughout. "
            "Conclude with a **References** section listing all cited papers."
        )
    }
    ]

    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.3)
    return response.strip(), input_tokens, output_tokens

def compute_final_score(comparisons: list) -> str:
    total = 0
    count = 0
    for comparison in comparisons:
        if comparison is None:
            continue
        comparison_data = comparison.get("comparison")
        if comparison_data is None:
            continue
        dims = comparison_data.get("dimension_comparisons", {})
        for dim_data in dims.values():
            score = dim_data.get("score")
            if isinstance(score, int):
                total += score
                count += 1
    score = (total * 10) / count if count > 0 else None
    return f"{score:.2f}"

def extract_bibliography(comparisons: list) -> str:
    """Extract all paperReference fields from comparisons and format as bibliography"""
    bibliography = []
    bibliography.append("Bibliography - References Used in Contribution Review")
    bibliography.append("=" * 50)
    bibliography.append("")
    
    # Extract unique paper references
    unique_references = set()
    for comparison in comparisons:
        if comparison is None:
            continue
        paper_ref = comparison.get("paperReference", "")
        if paper_ref and paper_ref not in unique_references:
            unique_references.add(paper_ref)
    
    sorted_references = sorted(list(unique_references), key=lambda x: x.split()[0].lower())
    
    for i, ref in enumerate(sorted_references, 1):
        bibliography.append(f"{i}. {ref}")
        bibliography.append("")
    
    bibliography.append(f"Total references: {len(sorted_references)}")
    
    return "\n".join(bibliography)

def save_bibliography(comparisons: list, output_path: str):
    """Save bibliography to a text file"""
    bibliography_content = extract_bibliography(comparisons)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bibliography_content)

def main():
    setup_logger()

    parser = argparse.ArgumentParser(description="Evaluate research plan contribution based on pairwise comparisons.")
    parser.add_argument("--research_plan", required=True, help="Path to the research plan .txt file.")
    parser.add_argument("--comparisons_file", required=True, help="Path to pairwise comparisons JSON file.")
    parser.add_argument("--llm_engine", required=True, help="LLM model name to use.")
    parser.add_argument("--litellm_name", help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514').")
    parser.add_argument("--output_file", required=True, help="Path to save final contribution evaluation.")
    parser.add_argument("--bibliography_file", required=False, help="Path to save bibliography/references txt file")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()

    logging.info("Loading research plan and pairwise comparisons...")
    research_plan_text = load_file(args.research_plan)
    comparisons = load_json(args.comparisons_file)

    llm = LLMEngine(
        llm_engine_name=args.llm_engine,
        api_key=os.environ.get("API_KEY"),  
        api_endpoint=os.environ.get("API_ENDPOINT")
    )
    
    llm_cost = model_cost[args.litellm_name] if args.litellm_name else None

    logging.info("Synthesizing final contribution assessment...")
    evaluation, input_tokens, output_tokens = prompt_final_synthesis(llm, research_plan_text, comparisons)
    
    if args.litellm_name:
        if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
            cost = 0
        else:
            cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                    llm_cost["output_cost_per_token"] * output_tokens)
        print(f"LLM cost for contribution_evaluator: ${cost:.6f}")
    else:
        cost = 0
        print(f"LLM tokens for contribution_evaluator: (Input: {input_tokens}, Output: {output_tokens})")

    if args.cost_log_file:
        cost_entry = {
            "step": "contribution_evaluator",
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')
    
    final_score = compute_final_score(comparisons)
    # full_output = f"{evaluation}\n\nFinal Score: {final_score}"
    full_output = evaluation
    
    bibliography_content = extract_bibliography(comparisons)
    if args.bibliography_file:
        save_bibliography(comparisons, args.bibliography_file)
        logging.info("Bibliography saved to %s", args.bibliography_file)
    
    final_output = full_output
    try:
        logging.info("Applying citation checking to contribution evaluation...")
        
        final_output = check_citations_from_strings(
            report_content=full_output,
            bibliography_content=bibliography_content
        )
        
        output_dir = os.path.dirname(args.output_file)
        citation_checked_path = os.path.join(output_dir, "contribution_citation_checked.md")
        with open(citation_checked_path, "w", encoding="utf-8") as f:
            f.write(final_output)
        logging.info("Citation-checked contribution evaluation saved to %s", citation_checked_path)
        
    except Exception as e:
        logging.warning(f"Citation checking failed with error: {e}")
        logging.info("Using original unchecked output as fallback.")
        final_output = full_output

    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    logging.info("Final contribution assessment saved to %s", args.output_file)

if __name__ == "__main__":
    main()
