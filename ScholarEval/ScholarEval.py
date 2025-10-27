import os
import sys
import json
import argparse
import subprocess
import time
from pathlib import Path


def run_command(cmd, working_dir=".", env=None):
    """Execute a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=working_dir, env=env)
    
    if result.returncode != 0:
        print(f"Command failed with return code {result.returncode}")
        print(f"STDERR: {result.stderr}")
        print(f"STDOUT: {result.stdout}")
        return False
    
    print("Command completed successfully")
    return True


def setup_environment():
    """Set up the environment for running the pipeline."""
    env = os.environ.copy()
    current_python_path = ":".join(sys.path)
    env["PYTHONPATH"] = f"{os.path.abspath('.')}:{current_python_path}:{env.get('PYTHONPATH', '')}"
    return env


def run_soundness_pipeline(research_plan_file, save_dir, cutoff_date, llm_engine_name, litellm_name):
    """Run the soundness evaluation pipeline."""
    print("\n=== Starting Soundness Pipeline ===")
    
    env = setup_environment()
    soundness_dir = Path(save_dir) / "soundness"
    soundness_dir.mkdir(exist_ok=True)
    
    # Copy research plan to soundness directory
    input_file = soundness_dir / "research_plan.txt"
    with open(research_plan_file, 'r') as src, open(input_file, 'w') as dst:
        dst.write(src.read())
    
    # Step 1: Extract methods
    print("\n1. Extracting methods from research plan...")
    cmd1 = [
        "python", "-m", "ScholarEval.soundness.extract_methods",
        "--input_file", str(input_file),
        "--output_file", str(soundness_dir / "methods.json"),
        "--llm_engine_name", llm_engine_name,
        "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")
    ]
    if litellm_name:
        cmd1.extend(["--litellm_name", litellm_name])
    if not run_command(cmd1, env=env):
        return False
    
    # Step 2: Make queries
    print("\n2. Generating search queries...")
    cmd2 = [
        "python", "-m", "ScholarEval.soundness.make_queries",
        "--research_plan", str(input_file),
        "--methods_file", str(soundness_dir / "methods.json"),
        "--output_file", str(soundness_dir / "queries.json"),
        "--llm_engine_name", llm_engine_name,
        "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")
    ]
    if litellm_name:
        cmd2.extend(["--litellm_name", litellm_name])
    if not run_command(cmd2, env=env):
        return False
    
    # Step 3: Snippet search
    print("\n3. Searching for relevant literature...")
    cmd3 = [
        "python", "-m", "ScholarEval.soundness.snippet_search",
        "--queries_file", str(soundness_dir / "queries.json"),
        "--methods_file", str(soundness_dir / "methods.json"),
        "--output_file", str(soundness_dir / "snippet"),
        "--pdf_dir", str(soundness_dir / "pdfs"),
        "--progress_file", str(soundness_dir / "snippet_progress.json")
    ]
    if cutoff_date:
        cmd3.extend(["--cutoff_date", cutoff_date])
    if not run_command(cmd3, env=env):
        return False
    
    # Step 4: Methods and results synthesis
    print("\n4. Analyzing existing methods in literature...")
    cmd4 = [
        "python", "-m", "ScholarEval.soundness.methods_and_results_synthesis",
        "--research_plan", str(input_file),
        "--methods_and_ref_file", str(soundness_dir / "snippet_references.json"),
        "--ref_and_paper_file", str(soundness_dir / "snippet_papers.json"),
        "--output_file", str(soundness_dir / "methods_analysis.json"),
        "--llm_engine_name", llm_engine_name,
        "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")
    ]
    if litellm_name:
        cmd4.extend(["--litellm_name", litellm_name])
    if not run_command(cmd4, env=env):
        return False
    
    # Step 5: Meta review
    print("\n5. Synthesizing soundness review...")
    cmd5 = [
        "python", "-m", "ScholarEval.soundness.meta_review",
        "--research_plan", str(input_file),
        "--mr_analysis_file", str(soundness_dir / "methods_analysis.json"),
        "--methods_and_ref_file", str(soundness_dir / "snippet_references.json"),
        "--output_file", str(soundness_dir / "meta_review.json"),
        "--markdown_output", str(soundness_dir / "meta_review.md"),
        "--llm_engine_name", llm_engine_name,
        "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")
    ]
    if litellm_name:
        cmd5.extend(["--litellm_name", litellm_name])
    if not run_command(cmd5, env=env):
        return False
    
    # Step 6: TL;DR soundness
    print("\n6. Generating TL;DR summary...")
    cmd6 = [
        "python", "-m", "ScholarEval.soundness.tldr_soundness",
        "--input_file", str(input_file),
        "--meta_review_file", str(soundness_dir / "meta_review.json"),
        "--llm_engine_name", llm_engine_name,
        "--output_file", str(soundness_dir / "tldr_soundness.txt"),
        "--markdown_file", str(soundness_dir / "tldr_soundness.md"),
        "--cost_log_file", str(soundness_dir / "soundness_costs.jsonl")
    ]
    if litellm_name:
        cmd6.extend(["--litellm_name", litellm_name])
    if not run_command(cmd6, env=env):
        return False
    
    print("\n=== Soundness Pipeline Completed ===")
    return True


def run_contribution_pipeline(research_plan_file, save_dir, cutoff_date, llm_engine_name, litellm_name):
    """Run the contribution evaluation pipeline."""
    print("\n=== Starting Contribution Pipeline ===")
    
    env = setup_environment()
    contribution_dir = Path(save_dir) / "contribution"
    contribution_dir.mkdir(exist_ok=True)
    
    # Copy research plan to contribution directory
    input_file = contribution_dir / "research_plan.txt"
    with open(research_plan_file, 'r') as src, open(input_file, 'w') as dst:
        dst.write(src.read())
    
    # Step 1: Extract dimensions and contributions
    print("\n1. Extracting dimensions and contributions...")
    cmd1 = [
        "python", "-m", "ScholarEval.contribution.extract_dimensions_and_contributions",
        "--input_file", str(input_file),
        "--llm_engine", llm_engine_name,
        "--output_file", str(contribution_dir / "dimensions_contributions.jsonl"),
        "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")
    ]
    if litellm_name:
        cmd1.extend(["--litellm_name", litellm_name])
    if not run_command(cmd1, env=env):
        return False
    
    # Step 2: Generate queries
    print("\n2. Generating contribution queries...")
    cmd2 = [
        "python", "-m", "ScholarEval.contribution.queries_generator",
        "--research_plan", str(input_file),
        "--contrib_file", str(contribution_dir / "dimensions_contributions.jsonl"),
        "--llm_engine_name", llm_engine_name,
        "--output_file", str(contribution_dir / "contribution_queries.json"),
        "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")
    ]
    if litellm_name:
        cmd2.extend(["--litellm_name", litellm_name])
    if not run_command(cmd2, env=env):
        return False
    
    # Step 3: Extract papers
    print("\n3. Extracting relevant papers...")
    cmd3 = [
        "python", "-m", "ScholarEval.contribution.paper_extractor",
        "--queries_file", str(contribution_dir / "contribution_queries.json"),
        "--output_file", str(contribution_dir / "contribution_papers.json"),
        "--progress_file", str(contribution_dir / "paper_progress.json")
    ]
    if cutoff_date:
        cmd3.extend(["--cutoff_date", cutoff_date])
    if not run_command(cmd3, env=env):
        return False
    
    # Step 4: Assess relevance
    print("\n4. Assessing paper relevance...")
    cmd4 = [
        "python", "-m", "ScholarEval.contribution.relevance_assessor",
        "--research_plan", str(input_file),
        "--papers_file", str(contribution_dir / "contribution_papers.json"),
        "--llm_engine", llm_engine_name,
        "--output_file", str(contribution_dir / "filtered_contribution_papers.json"),
        "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")
    ]
    if litellm_name:
        cmd4.extend(["--litellm_name", litellm_name])
    if not run_command(cmd4, env=env):
        return False
    
    # Step 5: Paper augmentation
    print("\n5. Augmenting papers with references...")
    cmd5 = [
        "python", "-m", "ScholarEval.contribution.paper_augmentation",
        "--relevant_papers", str(contribution_dir / "filtered_contribution_papers.json"),
        "--output_file", str(contribution_dir / "augmented_contribution_papers.json")
    ]
    if cutoff_date:
        cmd5.extend(["--cutoff_date", cutoff_date])
    if not run_command(cmd5, env=env):
        return False
    
    # Step 6: Embedding filter
    print("\n6. Applying embedding-based filtering...")
    cmd6 = [
        "python", "-m", "ScholarEval.contribution.embedding_filter",
        "--research_plan", str(input_file),
        "--papers_json", str(contribution_dir / "augmented_contribution_papers.json"),
        "--output", str(contribution_dir / "filtered_augmented_contribution_papers.json"),
        "--top_k", "100"
    ]
    if not run_command(cmd6, env=env):
        return False
    
    # Step 7: Re-assess relevance
    print("\n7. Re-assessing relevance of augmented papers...")
    cmd7 = [
        "python", "-m", "ScholarEval.contribution.relevance_assessor",
        "--research_plan", str(input_file),
        "--papers_file", str(contribution_dir / "filtered_augmented_contribution_papers.json"),
        "--llm_engine", llm_engine_name,
        "--output_file", str(contribution_dir / "final_contribution_papers.json"),
        "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")
    ]
    if litellm_name:
        cmd7.extend(["--litellm_name", litellm_name])
    if not run_command(cmd7, env=env):
        return False
    
    # Step 8: Paper sampling
    print("\n8. Sampling papers for comparison...")
    cmd8 = [
        "python", "-m", "ScholarEval.contribution.paper_sampler",
        "--input_file", str(contribution_dir / "final_contribution_papers.json"),
        "--output_file", str(contribution_dir / "sampled_final_contribution_papers.json")
    ]
    if not run_command(cmd8, env=env):
        return False
    
    # Step 9: Pairwise comparison
    print("\n9. Conducting pairwise comparisons...")
    cmd9 = [
        "python", "-m", "ScholarEval.contribution.pairwise_comparator",
        "--research_plan", str(input_file),
        "--papers_metadata", str(contribution_dir / "sampled_final_contribution_papers.json"),
        "--dimensions_file", str(contribution_dir / "dimensions_contributions.jsonl"),
        "--llm_engine", llm_engine_name,
        "--output_file", str(contribution_dir / "pairwise_comparisons.json"),
        "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")
    ]
    if litellm_name:
        cmd9.extend(["--litellm_name", litellm_name])
    if not run_command(cmd9, env=env):
        return False
    
    # Step 10: Prepare final context
    print("\n10. Preparing final contribution context...")
    cmd10 = [
        "python", "-m", "ScholarEval.contribution.prepare_final_contribution_context",
        "--input_file", str(contribution_dir / "pairwise_comparisons.json"),
        "--output_file", str(contribution_dir / "contribution_context.json")
    ]
    if not run_command(cmd10, env=env):
        return False
    
    # Step 11: Generate final review
    print("\n11. Synthesizing final contribution review...")
    cmd11 = [
        "python", "-m", "ScholarEval.contribution.contribution_review_synthesis",
        "--research_plan", str(input_file),
        "--comparisons_file", str(contribution_dir / "contribution_context.json"),
        "--llm_engine", llm_engine_name,
        "--output_file", str(contribution_dir / "contribution_review.txt"),
        "--cost_log_file", str(contribution_dir / "contribution_costs.jsonl")
    ]
    if litellm_name:
        cmd11.extend(["--litellm_name", litellm_name])
    if not run_command(cmd11, env=env):
        return False
    
    print("\n=== Contribution Pipeline Completed ===")
    return True


def main():
    parser = argparse.ArgumentParser(description="ScholarEval: Research Idea Evaluation Pipeline")
    parser.add_argument("--research_idea", required=True, help="Path to txt file containing research idea")
    parser.add_argument("--cutoff_date", help="Cutoff date for literature search (YYYY-MM-DD or 'none')")
    parser.add_argument("--llm_engine_name", required=True, help="LLM engine name")
    parser.add_argument("--save_to", required=True, help="Directory to save all intermediate files")
    parser.add_argument("--litellm_name", default="claude-sonnet-4-20250514", 
                        help="LiteLLM name for cost computation (default: claude-sonnet-4-20250514)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.research_idea):
        print(f"Error: Research idea file not found: {args.research_idea}")
        return 1
    
    # Handle cutoff date
    cutoff_date = None if not args.cutoff_date or args.cutoff_date.lower() == 'none' else args.cutoff_date
    
    # Create save directory
    save_dir = Path(args.save_to)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting ScholarEval pipeline...")
    print(f"Research idea: {args.research_idea}")
    print(f"Cutoff date: {cutoff_date or 'None'}")
    print(f"LLM engine: {args.llm_engine_name}")
    print(f"LiteLLM name: {args.litellm_name}")
    print(f"Save directory: {save_dir.absolute()}")
    
    start_time = time.time()
    
    # Run soundness pipeline
    if not run_soundness_pipeline(args.research_idea, save_dir, cutoff_date, 
                                  args.llm_engine_name, args.litellm_name):
        print("Soundness pipeline failed!")
        return 1
    
    # Run contribution pipeline
    if not run_contribution_pipeline(args.research_idea, save_dir, cutoff_date,
                                     args.llm_engine_name, args.litellm_name):
        print("Contribution pipeline failed!")
        return 1
    
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    print(f"\n=== ScholarEval Pipeline Completed Successfully ===")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Results saved to: {save_dir.absolute()}")
    print(f"  - Soundness results: {save_dir / 'soundness'}")
    print(f"  - Contribution results: {save_dir / 'contribution'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())