# steps/extract_methods.py
import os, json
import argparse
import pandas as pd
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils
from ..utils.citation_check import check_citations_from_strings

def compute_average_soundness_score(meta_review_data):
    """Compute the average of non-null soundness scores"""
    soundness_scores = []
    for method_data in meta_review_data.values():
        score = method_data.get('soundness_score')
        if score is not None:
            try:
                soundness_scores.append(float(score))
            except (ValueError, TypeError):
                continue
    return (sum(soundness_scores) / len(soundness_scores)) if soundness_scores else None

def main():
    parser = argparse.ArgumentParser(description="Aggregate literature evidence into a final evaluation")
    parser.add_argument("--input_file", required=True, help="Path to input research plan file (parquet or text)")
    parser.add_argument("--meta_review_file", required=True, help="Path to meta_reviews (JSON)")
    parser.add_argument("--output_file", required=True, help="Path to output JSON")
    parser.add_argument("--markdown_file", required=True, help="Path to output Markdown")
    parser.add_argument("--llm_engine_name", required=True, help="llm engine name (e.g., 'gpt-4o')")
    parser.add_argument("--litellm_name", required=True, help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514')")
    parser.add_argument("--bibliography_file", required=False, help="Path to bibliography file for citation checking")
    parser.add_argument("--cost_log_file", help="Path to centralized cost log file")
    args = parser.parse_args()

    API_KEY = os.environ.get("API_KEY_2")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    su = StringUtils()
    
    # Get LLM cost information
    llm_cost = model_cost[args.litellm_name]

    with open(args.input_file, "r", encoding="utf-8") as f:
        rp = f.read()
    with open(args.meta_review_file) as f:
        meta_review = json.load(f)["analysis"]

    # Compute average soundness score across method-level reviews
    avg_soundness_score = compute_average_soundness_score(meta_review)

    support_blocks, contra_blocks, suggestion_blocks, evaluation_blocks = [], [], [], []
    for i, (_, review) in enumerate(meta_review.items()):
        
        suggested_action = review.get('suggested_action', '').strip()
        support = review.get('support', '').strip()
        contradictions = review.get('contradictions', '').strip()
        evaluation = review.get('evaluation', '').strip()

        suggestion_blocks.append(f"[start suggestion {i}]\n{suggested_action}\n[end suggestion {i}]\n")
        support_blocks.append(f"[start support {i}]\n{support}\n[end support {i}]\n")
        contra_blocks.append(f"[start contradictions {i}]\n{contradictions}\n[end contradictions {i}]\n")
        evaluation_blocks.append(f"[start evaluation {i}]\n{evaluation}\n[end evaluation {i}]\n")

    support_text = "\n".join(support_blocks)
    contra_text = "\n".join(contra_blocks)
    suggestions_text = "\n".join(suggestion_blocks)
    evaluations_text = "\n".join(evaluation_blocks)

    prompt = [
        {
            "role": "user",
            "content": f"""You are an expert research assistant. You are given a research proposal and list of suggestions for improvement.

[start research proposal]
{rp}
[end research proposal]

{support_text}

{contra_text}

{suggestions_text}

Review the provided evidence that supports the methods in the research proposal. Please summarize the methods with the strongest support from existing literature.
Then review the provided evidence that contradicts the methods in the research proposal. Please summarize the methods with the strongest contradictions with existing literature.
Last, review the suggestions for improving the research proposal and synthesize 3 actionable, targeted, feasible suggestions for improvement.

It is required to copy the in-text citations with their links in markdown format [(author, YYYY-MM)](link) when you refer to text from that citation !!

JSON formatting requirements:
- Must be a complete, valid JSON object
- Start with an open bracket and end with closed bracket
- No trailing commas after the last property
- Validate JSON structure before output
```json
{{
    "strengths_summary": "brief summary of the strongest support from existing literature",
    "weaknesses_summary": "brief summary of the strongest contradictions with existing literature",
    "top_3_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}
```""",
        }
    ]

    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0.2)
    tldr = su.extract_json_output(response)

    # Calculate cost
    if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
        cost = 0
    else:
        cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                llm_cost["output_cost_per_token"] * output_tokens)
    
    print(f"LLM cost for tldr_soundness: ${cost:.6f}")

    # Attach average soundness score and cost information to final output JSON
    tldr['average_soundness_score'] = avg_soundness_score
    # Log cost to centralized cost log if provided
    if args.cost_log_file:
        cost_entry = {
            "step": "tldr_soundness",
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(tldr, f, indent=4, ensure_ascii=False)

    # Prepare Markdown summary
    strengths_md = "\n\n".join(tldr.get('strengths', [])) if isinstance(tldr.get('strengths'), list) else str(tldr.get('strengths', ''))
    weaknesses_md = "\n\n".join(tldr.get('weaknesses', [])) if isinstance(tldr.get('weaknesses'), list) else str(tldr.get('weaknesses', ''))
    suggestions_list = tldr.get('top_3_suggestions', [])
    suggestions_md = "\n\n".join(suggestions_list) if isinstance(suggestions_list, list) else str(suggestions_list)
    verdict_md = tldr.get('overall_verdict', '')

    score_text = f"{avg_soundness_score:.2f}" if avg_soundness_score is not None else "N/A (no valid scores found)"

    md = (
        f"## Final Evaluation — Overall Verdict\n{verdict_md}\n\n"
        f"## Strengths (most → least important)\n{strengths_md}\n\n"
        f"## Weaknesses (most → least critical)\n{weaknesses_md}\n\n"
        f"## Top 3 Suggestions (priority order)\n{suggestions_md}\n\n"
        f"## Average Soundness Score (from prior step)\n{score_text}"
    )
    # Apply citation checking if bibliography file is provided
    final_output = md
    if args.bibliography_file and os.path.exists(args.bibliography_file):
        try:
            # Read bibliography content
            with open(args.bibliography_file, "r", encoding="utf-8") as f:
                bibliography_content = f.read()
            
            # Apply citation checking
            print("Applying citation checking to soundness review...")
            final_output = check_citations_from_strings(
                report_content=md,
                bibliography_content=bibliography_content
            )
            
            # Save citation-checked version as the primary output
            output_dir = os.path.dirname(args.output_file)
            citation_checked_path = os.path.join(output_dir, "tldr_citation_checked.md")
            with open(citation_checked_path, "w", encoding="utf-8") as f:
                f.write(final_output)
            print(f"Citation-checked soundness review saved to {citation_checked_path}")
            
        except Exception as e:
            print(f"Warning: Citation checking failed with error: {e}")
            print("Using original unchecked output as fallback.")
            final_output = md
    
    # Save final output to the specified output file (either citation-checked or original)
    with open(args.output_file, "w", encoding="utf-8") as f:
        f.write(final_output)
    
    # Also save to markdown file for backward compatibility
    with open(args.markdown_file, "w", encoding="utf-8") as f:
        f.write(final_output)

if __name__ == '__main__':
    main()
