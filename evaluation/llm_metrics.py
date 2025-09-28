import json
import argparse
import csv
import random
from pathlib import Path
from litellm import model_cost
from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils
import os

def read_responses_from_subfolders(folder_path):
    """Read combined.txt files from subfolders and return contents with subfolder names."""
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    responses = {}
    
    # Find all subfolders that contain combined.txt
    for subfolder in folder.iterdir():
        if subfolder.is_dir():
            combined_file = subfolder / "combined.txt"
            if combined_file.exists():
                print(f"Reading response from: {subfolder.name}")
                with open(combined_file, 'r', encoding='utf-8') as f:
                    responses[subfolder.name] = f.read().strip()
            else:
                print(f"Warning: No combined.txt found in {subfolder.name}")
    
    return responses

def align_responses(responses_a_dict, responses_b_dict):
    """Align responses from both folders based on matching subfolder names."""
    aligned_a = []
    aligned_b = []
    aligned_names = []
    
    # Find common subfolder names
    common_names = set(responses_a_dict.keys()) & set(responses_b_dict.keys())
    
    # Sort for consistent ordering
    common_names = sorted(common_names)
    
    print(f"Found {len(common_names)} matching subfolders:")
    for name in common_names:
        print(f"  - {name}")
        aligned_a.append(responses_a_dict[name])
        aligned_b.append(responses_b_dict[name])
        aligned_names.append(name)
    
    print(f"Total aligned pairs: {len(aligned_a)}")
    
    return aligned_a, aligned_b, aligned_names

def parse_judgment_response(response_text: str) -> dict:
    """
    Parse the LLM's JSON response to extract rationales and winners for each criterion.
    
    Returns:
        dict: Parsed judgment with rationales and winners, or None if parsing fails
    """
    su = StringUtils()
    
    try:
        # Extract JSON from the response
        judgment_json = su.extract_json_output(response_text)
        
        # Validate required fields
        required_fields = [
            "Evidence-based-rationale", "Evidence-based-winner",
            "Actionability-rationale", "Actionability-winner", 
            "Depth-rationale", "Depth-winner"
        ]
        
        for field in required_fields:
            if field not in judgment_json:
                print(f"Warning: Missing field {field} in judgment response")
                return None
        
        # Normalize winners to A, B, or Tie
        for criterion in ["Evidence-based", "Actionability", "Depth"]:
            winner_key = f"{criterion}-winner"
            winner = str(judgment_json[winner_key]).strip().upper()
            if winner not in ['A', 'B', 'TIE']:
                if 'TIE' in winner:
                    winner = 'TIE'
                elif 'A' in winner:
                    winner = 'A'
                elif 'B' in winner:
                    winner = 'B'
                else:
                    winner = 'UNCLEAR'
            judgment_json[winner_key] = winner
        
        return judgment_json
    
    except Exception as e:
        print(f"Error parsing judgment response: {e}")
        print(f"Response text: {response_text}")
        return None

def judge_reports(llm: LLMEngine, original_report_a: str, original_report_b: str) -> tuple:
    """
    Prompts the LLM to judge between two reports and choose A or B.
    Randomizes the order to reduce position bias.
    
    Returns:
        tuple: (response, input_tokens, output_tokens, is_swapped)
        is_swapped: True if original A became B in the prompt, False otherwise
    """
    # Randomly decide whether to swap the order
    is_swapped = random.choice([True, False])
    
    if is_swapped:
        # Original A becomes Report B, Original B becomes Report A
        prompt_report_a = original_report_b
        prompt_report_b = original_report_a
    else:
        # Keep original order
        prompt_report_a = original_report_a
        prompt_report_b = original_report_b
    
    prompt = [
        {"role": "system", "content": (
            "You are an expert scientific reviewer. You will be given two research evaluation reports (Report A and Report B) "
            "and you must answer which one is better based on the following criteria:"
            "Evidence-based: An evidence-based report is one that grounds its claims in the literature and backs them with relevant citations that improve traceability and its overall trustworthiness and reliability. Which report is more evidence-based?"
            "Actionability: Actionability is the extent to which the report offers varied, clear, and actionable suggestions that are likely to improve the research idea. Which report is more actionable?"
            "Depth: Depth is measured by the degree of engagement with the point being evaluated and the literature cited. A deep report discusses each point from multiple angles and references specific details about the literature it cites, rather than relying on generic statements followed by citations. Which response has greater depth?"
            "You must choose either A, B, or Tie (when both responses are equivalent). Read both responses thoroughly before judging, and give thorough rationales that show that you are a fair and unbiased judge. If you choose that one response is better, clearly rationalize why. If they are equivalent in quality based on any of these criteria then you nned to choose Tie."
        )},
        {"role": "user", "content": (
            f"Report A:\n{prompt_report_a}\n\n"
            f"Report B:\n{prompt_report_b}\n\n"
            f"Judge the responses based on the criteria, while respecting the output format below."
            "```json\n"
            "{{\n"
            "  \"Evidence-based-rationale\": ... "
            "  \"Evidence-based-winner\": <A, B, or Tie> "
            "  \"Actionability-rationale\": ... "
            "  \"Actionability-winner\": <A, B, or Tie> "
            "  \"Depth-rationale\": ... "
            "  \"Depth-winner\": <A, B, or Tie> "
            "}}\n"
            "```"
        )}
    ]
    
    response, input_tokens, output_tokens = llm.respond(prompt, temperature=0)
    return response, input_tokens, output_tokens, is_swapped

def correct_winners_for_swap(parsed_judgment: dict, is_swapped: bool) -> dict:
    """
    Correct the winners in the parsed judgment based on whether the order was swapped.
    
    Args:
        parsed_judgment: The parsed judgment dict
        is_swapped: True if original A became B in the prompt
        
    Returns:
        dict: Corrected judgment with winners mapped back to original positions
    """
    if not parsed_judgment or not is_swapped:
        return parsed_judgment
    
    corrected = parsed_judgment.copy()
    
    # If swapped, A in the prompt corresponds to original B, and B corresponds to original A
    for criterion in ["Evidence-based", "Actionability", "Depth"]:
        winner_key = f"{criterion}-winner"
        if winner_key in corrected:
            prompt_winner = corrected[winner_key]
            if prompt_winner == 'A':
                # Prompt A was original B
                corrected[winner_key] = 'B'
            elif prompt_winner == 'B':
                # Prompt B was original A
                corrected[winner_key] = 'A'
            # TIE and UNCLEAR stay unchanged
    
    return corrected


def main():
    parser = argparse.ArgumentParser(description="Compare two sets of reports using LLM judge with alignment.")
    parser.add_argument("--responses_a", required=True, help="Path to folder containing subfolders with combined.txt files (Report A).")
    parser.add_argument("--responses_b", required=True, help="Path to folder containing subfolders with combined.txt files (Report B).")
    parser.add_argument("--output", required=True, help="Output folder for results.")
    parser.add_argument("--llm_engine_name", required=True, help="Name of the LLM engine to use (e.g., 'gpt-4o').")
    parser.add_argument("--litellm_name", required=True, help="LiteLLM model name for cost calculation (e.g., 'claude-sonnet-4-20250514').")
    parser.add_argument("--cost_log_file",default = "agent/benchmark_creation/llm_judge_cost_file.log", help="Path to centralized cost log file")
    parser.add_argument("--random_seed", type=int, help="Random seed for reproducible order randomization")
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.random_seed is not None:
        random.seed(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")

    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM engine
    API_KEY = os.environ.get("API_KEY_1")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    su = StringUtils()
    
    # Get LLM cost information
    llm_cost = model_cost[args.litellm_name]

    # Read responses from both folders
    print("Reading responses A...")
    responses_a_dict = read_responses_from_subfolders(args.responses_a)
    
    print("Reading responses B...")
    responses_b_dict = read_responses_from_subfolders(args.responses_b)
    
    # Align responses based on matching subfolder names
    aligned_a, aligned_b, aligned_names = align_responses(responses_a_dict, responses_b_dict)
    
    if len(aligned_a) == 0:
        print("Error: No matching subfolders found between the two response folders!")
        return
    
    # Process each aligned pair
    total_cost = 0
    total_input_tokens = 0
    total_output_tokens = 0
    judgments = []
    parsed_judgments = []
    
    print("Running LLM judgments...")
    for i, (report_a, report_b, name) in enumerate(zip(aligned_a, aligned_b, aligned_names)):
        print(f"Processing {i+1}/{len(aligned_a)}: {name}")
        
        # Perform judgment with randomization
        judgment_text, input_tokens, output_tokens, is_swapped = judge_reports(llm, report_a, report_b)
        
        # Calculate cost for this judgment
        if args.litellm_name == "meta_llama/Llama-3.3-70B-Instruct":
            cost = 0
        else:
            cost = (llm_cost["input_cost_per_token"] * input_tokens + 
                    llm_cost["output_cost_per_token"] * output_tokens)
        
        total_cost += cost
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        
        # Parse the judgment response
        parsed_judgment = parse_judgment_response(judgment_text)
        
        # Correct winners if order was swapped
        corrected_judgment = correct_winners_for_swap(parsed_judgment, is_swapped)
        
        judgments.append(judgment_text)
        parsed_judgments.append(corrected_judgment)
    
    print(f"Total LLM cost for all judgments: ${total_cost:.6f}")
    
    # Log total cost to centralized cost log if provided
    if args.cost_log_file:
        cost_entry = {
            "step": "llm_judge_batch",
            "cost": total_cost,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "num_comparisons": len(aligned_a)
        }
        with open(args.cost_log_file, 'a') as f:
            json.dump(cost_entry, f)
            f.write('\n')
    
    # Save results to CSV file
    csv_path = output_dir / "results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'Subfolder_Name', 'Raw_Judgment',
            'Evidence_Rationale', 'Evidence_Winner',
            'Actionability_Rationale', 'Actionability_Winner', 
            'Depth_Rationale', 'Depth_Winner'
        ])
        
        for name, judgment, parsed in zip(aligned_names, judgments, parsed_judgments):
            if parsed:
                writer.writerow([
                    name, judgment,
                    parsed.get('Evidence-based-rationale', ''),
                    parsed.get('Evidence-based-winner', 'UNCLEAR'),
                    parsed.get('Actionability-rationale', ''),
                    parsed.get('Actionability-winner', 'UNCLEAR'),
                    parsed.get('Depth-rationale', ''),
                    parsed.get('Depth-winner', 'UNCLEAR')
                ])
            else:
                writer.writerow([name, judgment, '', 'UNCLEAR', '', 'UNCLEAR', '', 'UNCLEAR'])
    
    print(f"Results saved to {csv_path}")

    # Calculate statistics for each criterion
    criteria = ["Evidence-based", "Actionability", "Depth"]
    criterion_stats = {}
    
    for criterion in criteria:
        winner_key = f"{criterion}-winner"
        winners = []
        
        for parsed in parsed_judgments:
            if parsed and winner_key in parsed:
                winners.append(parsed[winner_key])
            else:
                winners.append('UNCLEAR')
        
        a_wins = sum(1 for winner in winners if winner == 'A')
        b_wins = sum(1 for winner in winners if winner == 'B')
        ties = sum(1 for winner in winners if winner == 'TIE')
        unclear = sum(1 for winner in winners if winner == 'UNCLEAR')
        total = len(winners)
        
        criterion_stats[criterion] = {
            'total_comparisons': total,
            'a_wins': a_wins,
            'b_wins': b_wins,
            'ties': ties,
            'unclear': unclear,
            'a_win_rate': (a_wins / total) * 100 if total > 0 else 0,
            'b_win_rate': (b_wins / total) * 100 if total > 0 else 0,
            'tie_rate': (ties / total) * 100 if total > 0 else 0
        }
    
    # Save summary statistics to text file
    summary_path = output_dir / "summary_statistics.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("LLM Judge Summary Statistics (Multi-Criteria Comparison)\n")
        f.write("=" * 60 + "\n\n")
        
        total_comparisons = len(parsed_judgments)
        f.write(f"Total comparisons: {total_comparisons}\n\n")
        
        # Overall statistics across all criteria
        f.write("Overall Statistics Across All Criteria:\n")
        f.write("-" * 40 + "\n")
        all_a_wins = sum(stats['a_wins'] for stats in criterion_stats.values())
        all_b_wins = sum(stats['b_wins'] for stats in criterion_stats.values())
        all_ties = sum(stats['ties'] for stats in criterion_stats.values())
        all_unclear = sum(stats['unclear'] for stats in criterion_stats.values())
        all_total = all_a_wins + all_b_wins + all_ties + all_unclear
        
        f.write(f"Response A wins: {all_a_wins}/{all_total} ({(all_a_wins/all_total)*100:.1f}%)\n")
        f.write(f"Response B wins: {all_b_wins}/{all_total} ({(all_b_wins/all_total)*100:.1f}%)\n")
        f.write(f"Ties: {all_ties}/{all_total} ({(all_ties/all_total)*100:.1f}%)\n")
        f.write(f"Unclear results: {all_unclear}/{all_total} ({(all_unclear/all_total)*100:.1f}%)\n\n")
        
        # Per-criterion statistics
        for criterion in criteria:
            stats = criterion_stats[criterion]
            f.write(f"{criterion} Criterion:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Response A wins: {stats['a_wins']} ({stats['a_win_rate']:.1f}%)\n")
            f.write(f"Response B wins: {stats['b_wins']} ({stats['b_win_rate']:.1f}%)\n")
            f.write(f"Ties: {stats['ties']} ({stats['tie_rate']:.1f}%)\n")
            f.write(f"Unclear results: {stats['unclear']}\n\n")
        
        # Detailed breakdown by subfolder
        f.write("Detailed Results by Subfolder:\n")
        f.write("-" * 30 + "\n")
        for name, parsed in zip(aligned_names, parsed_judgments):
            f.write(f"{name}:\n")
            if parsed:
                for criterion in criteria:
                    winner = parsed.get(f"{criterion}-winner", "UNCLEAR")
                    f.write(f"  {criterion}: {winner}\n")
            else:
                f.write("  Parse error - all criteria: UNCLEAR\n")
            f.write("\n")
    
    print(f"Summary statistics saved to {summary_path}")
    
    return criterion_stats


if __name__ == "__main__":
    main()