# Absolute Grading: Outputs score of 1 to 5
import os
import sys
import argparse
import csv
import statistics
import json
from pathlib import Path
from prometheus_eval.litellm import LiteLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from litellm.exceptions import RateLimitError

def read_research_plans_and_references(folder_path, target_folder_names):
    """Read research plans and reference statements from specific subfolders.
    Each subfolder contains <folder_name>.txt, <folder_name>.jsonl, and cutoff.txt
    Only reads from folders specified in target_folder_names.
    Returns individual statements with their metadata for statement-level evaluation."""
    statement_data = []
    valid_folder_names = []
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    for folder_name in target_folder_names:
        subdir = folder / folder_name
        if not subdir.exists() or not subdir.is_dir():
            print(f"Warning: Skipping {folder_name} - subfolder not found in plans_and_references")
            continue
        
        # Read research plan from <folder_name>.txt
        research_plan_file = subdir / f"{folder_name}.txt"
        if not research_plan_file.exists():
            print(f"Warning: Skipping {folder_name} - research plan file not found: {research_plan_file}")
            continue
        
        # Read reference answer from <folder_name>.jsonl
        reference_file = subdir / f"{folder_name}.jsonl"
        if not reference_file.exists():
            print(f"Warning: Skipping {folder_name} - reference answer file not found: {reference_file}")
            continue
        
        # Read research plan
        with open(research_plan_file, 'r', encoding='utf-8') as f:
            research_plan = f.read().strip()
            print(f"Read research plan from: {research_plan_file}")
        
        # Read individual statements with metadata
        statements_found = False
        with open(reference_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'statement' in data:
                            statement_info = {
                                'folder_name': folder_name,
                                'research_plan': research_plan,
                                'statement': data['statement'],
                                'type': data.get('type', 'unknown'),
                                'axis': data.get('axis', 'unknown'),
                                'severity': data.get('severity', 'unknown')
                            }
                            statement_data.append(statement_info)
                            statements_found = True
                        else:
                            print(f"Warning: No 'statement' field found in line {line_num} of {reference_file}")
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON in line {line_num} of {reference_file}: {e}")
        
        if statements_found:
            valid_folder_names.append(folder_name)
            print(f"Read {len([s for s in statement_data if s['folder_name'] == folder_name])} statements from: {reference_file}")
    
    return statement_data, valid_folder_names

def get_available_response_folder_names(folder_path):
    """Get list of folder names that have valid responses (contain combined.txt)."""
    available_folders = []
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Get all subdirectories and sort alphabetically
    subdirs = [d for d in folder.iterdir() if d.is_dir()]
    subdirs = sorted(subdirs, key=lambda x: x.name)
    
    for subdir in subdirs:
        combined_file = subdir / "combined.txt"
        if combined_file.exists():
            available_folders.append(subdir.name)
            print(f"Found valid response in: {subdir.name}")
        else:
            print(f"Skipping {subdir.name} - no combined.txt file found")
    
    return available_folders

def read_responses_from_subfolders(folder_path, folder_names):
    """Read responses from subfolders, each containing combined.txt."""
    responses = []
    
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder {folder_path} does not exist")
    
    # Read responses in the order of provided folder names
    for folder_name in folder_names:
        subdir = folder / folder_name
        combined_file = subdir / "combined.txt"
        
        with open(combined_file, 'r', encoding='utf-8') as f:
            response = f.read().strip()
            responses.append(response)
            print(f"Read response from: {combined_file}")
    
    return responses

def main():
    parser = argparse.ArgumentParser(description='Prometheus Judge for evaluating research plan reviews at statement level')
    parser.add_argument('--plans_and_references', required=True, help='Path to folder containing subfolders with research plans and reference answers')
    parser.add_argument('--responses', required=True, help='Path to folder containing subfolders with response combined.txt files')
    parser.add_argument('--output', required=True, help='Output folder for results')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # First, get available response folder names (this determines what we can evaluate)
    print("Scanning for available responses...")
    available_response_folders = get_available_response_folder_names(args.responses)
    print(f"Found {len(available_response_folders)} folders with valid responses")
    
    # Then, read research plans and reference statements only for folders that have responses
    print("Reading matching research plans and references...")
    statement_data, valid_folder_names = read_research_plans_and_references(
        args.plans_and_references, available_response_folders
    )
    
    # Read responses for valid folders
    print("Reading responses for valid folders...")
    folder_responses = {}
    for folder_name in valid_folder_names:
        responses = read_responses_from_subfolders(args.responses, [folder_name])
        folder_responses[folder_name] = responses[0]
    
    # Create statement-level triplets (research_plan, response, statement)
    print("Creating statement-level evaluation triplets...")
    evaluation_triplets = []
    for stmt_data in statement_data:
        folder_name = stmt_data['folder_name']
        if folder_name in folder_responses:
            triplet = {
                'folder_name': folder_name,
                'research_plan': stmt_data['research_plan'],
                'response': folder_responses[folder_name],
                'statement': stmt_data['statement'],
                'type': stmt_data['type'],
                'axis': stmt_data['axis'],
                'severity': stmt_data['severity']
            }
            evaluation_triplets.append(triplet)
    
    print(f"Created {len(evaluation_triplets)} statement-level evaluation triplets")
    if len(available_response_folders) > len(valid_folder_names):
        skipped_count = len(available_response_folders) - len(valid_folder_names)
        print(f"Note: {skipped_count} response folders were skipped due to missing research plans or references")
    
    # Prepare data for Prometheus evaluation
    instructions = []
    responses = []
    reference_answers = []
    
    instruction_template = """Given the following research plan detailing the problem, methodology, and planned experiments that a research team intends to implement, give an evaluation of the research plan based on its technical soundness and scientific novelty and contribution to the field.

    Research plan: {research_plan}"""
    
    for triplet in evaluation_triplets:
        instructions.append(instruction_template.format(research_plan=triplet['research_plan']))
        responses.append(triplet['response'])
        reference_answers.append(triplet['statement'])

    model = LiteLLM('openai/gpt-4o')
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    rubric_data = {
        "criteria": "You are given a single essential statement from a research idea review targeting the technical soundness and scientific contribution of a research idea. This statement was extracted from actual human reviews about the research idea and will be used as your ground-truth reference. Additionally, you will be given a research idea review that aims to give a detailed evaluation of the same research idea. The review given to you will typically be long and composed of multiple sections and references to related papers. Your job is to verify that the given review has a mention of the reference statement in its content. These references can be paraphrased of articulated differently, but they should express the same meaning as the reference statement. Does the given review express the reference statement?",
        "score1_description": "The given review does not mention the reference statement at all",
        "score2_description": "The given review mentions some related concepts but does not express the core meaning of the reference statement",
        "score3_description": "The given review partially expresses the reference statement but misses key aspects or nuances",
        "score4_description": "The given review expresses most aspects of the reference statement with minor omissions or slight differences in emphasis",
        "score5_description": "The given review clearly expresses the reference statement, capturing its core meaning and implications",
    }
    # rubric_data = {
    #     "criteria": "You are given a single statement from a research idea review targeting the technical soundness an d/or scientific contribution of a research idea. This statement was extracted from actual human reviews about the research idea and will be used as your reference. Additionally, you will be given a research idea review that aims to give a detailed evaluation of the same research idea. The review given to you will typically be long and composed of multiple sections and references to related papers. Your job is to look whether the given review has some kind of reference or insinuation of the reference statement somewhere in its content. The statement does not have to be mentioned clearly, so only check whether there is some kind of reference to it that conveys a similar meaning. Does the given review have some mention of the reference statement?",
    #     "score1_description": "The given review does not have the slightest mention to concepts related to the reference statement anywhere in its content. It's talking about completely different things.",
    #     "score2_description": "The given review has a negligeable mention to concepts touching on the reference statement, but this can be missed. ",
    #     "score3_description": "The given review touches on themes and background ideas related to the reference statement, but does not mention it directly.",
    #     "score4_description": "The given review has mentions (direct or indirect) to most of the concepts related to the reference statement, but not fully.",
    #     "score5_description": "The given review captures the main meaning of the reference statement clearly.",
    # }
 
    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    print("Running Prometheus evaluation for individual statements...")
    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=responses,
        rubric=score_rubric,
        reference_answers=reference_answers
    )

    # Save feedbacks and scores to CSV file with statement metadata
    csv_path = output_dir / "results.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Instance_ID', 'Statement', 'Type', 'Axis', 'Severity', 'Feedback', 'Score'])
        
        for i, (triplet, feedback, score) in enumerate(zip(evaluation_triplets, feedbacks, scores)):
            writer.writerow([
                f"{triplet['folder_name']}_stmt_{i+1}",
                triplet['statement'],
                triplet['type'],
                triplet['axis'], 
                triplet['severity'],
                feedback,
                score
            ])
    
    print(f"Results saved to {csv_path}")

    # Save summary statistics to text file
    numeric_scores = [int(score) for score in scores]
    
    summary_stats = {
        'mean': statistics.mean(numeric_scores),
        'median': statistics.median(numeric_scores),
        'stdev': statistics.stdev(numeric_scores) if len(numeric_scores) > 1 else 0,
        'min': min(numeric_scores),
        'max': max(numeric_scores),
        'count': len(numeric_scores),
        'total_folders': len(valid_folder_names),
        'total_statements': len(evaluation_triplets)
    }
    
    summary_path = output_dir / "summary_statistics.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Prometheus Judge Summary Statistics (Statement Level)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total folders evaluated: {summary_stats['total_folders']}\n")
        f.write(f"Total statements evaluated: {summary_stats['total_statements']}\n")
        f.write(f"Mean score: {summary_stats['mean']:.2f}\n")
        f.write(f"Median score: {summary_stats['median']:.1f}\n")
        f.write(f"Standard deviation: {summary_stats['stdev']:.2f}\n")
        f.write(f"Minimum score: {summary_stats['min']}\n")
        f.write(f"Maximum score: {summary_stats['max']}\n")
    
    print(f"Summary statistics saved to {summary_path}")
    
    return summary_stats

if __name__ == "__main__":
    main()


