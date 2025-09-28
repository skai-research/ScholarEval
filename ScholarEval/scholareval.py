#!/usr/bin/env python3
"""
Simple ScholarEval Pipeline
Takes a single research plan and runs both soundness and contribution evaluation workflows.
"""

import os
import sys
import argparse
import json
import time
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple, Optional

def setup_logging(output_dir: Path) -> logging.Logger:
    """Setup logging for the ScholarEval pipeline."""
    log_file = output_dir / "scholareval_processing.log"
    
    # Create logger
    logger = logging.getLogger("scholareval")
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def run_soundness_workflow(input_file: Path, output_dir: Path, llm_engine_name: str, litellm_name: str,
                          logger: logging.Logger) -> Tuple[bool, str, Dict]:
    """Run the soundness workflow for a research plan."""
    try:
        logger.info("Starting soundness workflow...")
        
        # Set up centralized cost logging
        soundness_cost_log = output_dir / "soundness_costs.jsonl"
        
        # Set up environment
        env = os.environ.copy()
        working_dir = Path(__file__).parent  # Current ScholarEval directory
        current_python_path = ":".join(sys.path)
        env["PYTHONPATH"] = f"{os.path.abspath(working_dir)}:{current_python_path}:{env.get('PYTHONPATH', '')}"
        
        # Step 1: Extract methods
        logger.info("Step 1: Extracting methods from research plan...")
        cmd1 = ["python", "-m", "soundness.extract_methods", 
                "--input_file", str(input_file), 
                "--output_file", str(output_dir / "methods.json"), 
                "--llm_engine_name", "GPT-4.1-nano",
                "--litellm_name", litellm_name,
                "--cost_log_file", str(soundness_cost_log)]
        result = subprocess.run(cmd1, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Method extraction failed: {result.stderr}")
            return False, f"Method extraction failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Methods extracted successfully!")

        # Step 2: Make queries
        logger.info("Step 2: Generating search queries...")
        cmd2 = ["python", "-m", "soundness.make_queries", 
                "--research_plan", str(input_file),
                "--methods_file", str(output_dir / "methods.json"), 
                "--output_file", str(output_dir / "queries.json"), 
                "--llm_engine_name", "GPT-4.1-nano",
                "--litellm_name", litellm_name,
                "--cost_log_file", str(soundness_cost_log)]
        result = subprocess.run(cmd2, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Query generation failed: {result.stderr}")
            return False, f"Query generation failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Search queries generated successfully!")

        # Step 3: Snippet search
        logger.info("Step 3: Searching for relevant literature...")
        progress_file = output_dir / "snippet_progress.json"
        cmd3 = ["python", "-m", "soundness.snippet_search", 
                "--queries_file", str(output_dir / "queries.json"), 
                "--methods_file", str(output_dir / "methods.json"), 
                "--output_file", str(output_dir / "snippet"), 
                "--pdf_dir", str(output_dir / "pdfs"), 
                "--progress_file", str(progress_file)]
        
        result = subprocess.run(cmd3, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Snippet search failed: {result.stderr}")
            return False, f"Snippet search failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Literature search completed successfully!")

        # Step 4: Compare methods
        logger.info("Step 4: Analyzing existing methods in the literature...")
        cmd4 = ["python", "-m", "soundness.methods_and_results_synthesis", 
                "--research_plan", str(input_file),
                "--methods_and_ref_file", str(output_dir / "snippet_references.json"), 
                "--ref_and_paper_file", str(output_dir / "snippet_papers.json"), 
                "--output_file", str(output_dir / "methods_analysis.json"), 
                "--llm_engine_name", llm_engine_name,
                "--litellm_name", litellm_name,
                "--cost_log_file", str(soundness_cost_log)]
        result = subprocess.run(cmd4, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Method analysis failed: {result.stderr}")
            return False, f"Method analysis failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Method analysis completed!")

        # Step 5: Synthesize review
        logger.info("Step 5: Synthesizing method-level soundness review...")
        review_file = output_dir / "meta_review.json"
        cmd5 = ["python", "-m", "soundness.meta_review", 
                "--research_plan", str(input_file),
                "--mr_analysis_file", str(output_dir / "methods_analysis.json"), 
                "--methods_and_ref_file", str(output_dir / "snippet_references.json"), 
                "--output_file", str(review_file),
                "--markdown_output", str(output_dir / "meta_review.md"), 
                "--llm_engine_name", llm_engine_name,
                "--litellm_name", litellm_name,
                "--cost_log_file", str(soundness_cost_log)]
        result = subprocess.run(cmd5, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Meta review failed: {result.stderr}")
            return False, f"Meta review failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Meta review completed!")

        # Step 6: Generate TL;DR soundness summary
        logger.info("Step 6: Generating TL;DR soundness summary...")
        tldr_file = output_dir / "tldr_soundness.txt"
        cmd6 = ["python", "-m", "soundness.tldr_soundness", 
                "--research_plan", str(input_file),
                "--meta_review_file", str(output_dir / "meta_review.md"), 
                "--llm_engine_name", llm_engine_name,
                "--litellm_name", litellm_name,
                "--output_file", str(tldr_file),
                "--cost_log_file", str(soundness_cost_log)]
        result = subprocess.run(cmd6, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"TL;DR soundness summary failed: {result.stderr}")
            return False, f"TL;DR soundness summary failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("TL;DR soundness summary completed!")

        # Collect cost information
        cost_info = {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        
        if soundness_cost_log.exists():
            try:
                with open(soundness_cost_log, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            step_data = json.loads(line)
                            cost_info['total_cost'] += step_data.get('cost', 0)
                            cost_info['total_input_tokens'] += step_data.get('input_tokens', 0)
                            cost_info['total_output_tokens'] += step_data.get('output_tokens', 0)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not read cost log {soundness_cost_log}: {e}")
        
        logger.info(f"Soundness workflow total cost: ${cost_info['total_cost']:.4f}")

        # Read the final output (TL;DR summary)
        if tldr_file.exists():
            with open(tldr_file, 'r') as f:
                soundness_output = f.read()
            logger.info("Soundness workflow completed successfully!")
            return True, soundness_output, cost_info
        else:
            logger.error("TL;DR soundness summary file not found!")
            return False, "TL;DR soundness summary file not found!", cost_info

    except Exception as e:
        logger.error(f"Soundness workflow failed: {e}")
        return False, f"Soundness workflow failed: {e}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}

def run_contribution_workflow(input_file: Path, output_dir: Path, llm_engine_name: str, litellm_name: str,
                             logger: logging.Logger) -> Tuple[bool, str, Dict]:
    """Run the contribution workflow for a research plan."""
    try:
        logger.info("Starting contribution workflow...")
        
        # Set up centralized cost logging
        contribution_cost_log = output_dir / "contribution_costs.jsonl"
        
        # Set up environment
        env = os.environ.copy()
        working_dir = Path(__file__).parent  # Current ScholarEval directory
        current_python_path = ":".join(sys.path)
        env["PYTHONPATH"] = f"{os.path.abspath(working_dir)}:{current_python_path}:{env.get('PYTHONPATH', '')}"

        # Step 1: Extract dimensions and contributions
        logger.info("Step 1: Extracting dimensions and contributions...")
        cmd1 = ["python", "-m", "contribution.extract_dimensions_and_contributions", 
                "--input_file", str(input_file), 
                "--llm_engine", "GPT-4.1-nano",
                "--litellm_name", litellm_name,
                "--output_file", str(output_dir / "dimensions_contributions.jsonl"),
                "--cost_log_file", str(contribution_cost_log)]
        result = subprocess.run(cmd1, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Dimension extraction failed: {result.stderr}")
            return False, f"Dimension extraction failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Dimensions and contributions extracted successfully!")

        # Step 2: Generate contribution queries
        logger.info("Step 2: Generating targeted search queries...")
        cmd2 = ["python", "-m", "contribution.queries_generator", 
                "--research_plan", str(input_file), 
                "--contrib_file", str(output_dir / "dimensions_contributions.jsonl"), 
                "--llm_engine_name", "GPT-4.1-nano",
                "--litellm_name", litellm_name,
                "--output_file", str(output_dir / "contribution_queries.json"),
                "--cost_log_file", str(contribution_cost_log)]
        result = subprocess.run(cmd2, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Query generation failed: {result.stderr}")
            return False, f"Query generation failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Search queries generated successfully!")

        # Step 3: Extract papers
        logger.info("Step 3: Extracting relevant papers from Semantic Scholar...")
        progress_file = output_dir / "paper_progress.json"
        cmd3 = ["python", "-m", "contribution.paper_extractor", 
                "--queries_file", str(output_dir / "contribution_queries.json"), 
                "--output_file", str(output_dir / "contribution_papers.json"), 
                "--progress_file", str(progress_file)]
        
        result = subprocess.run(cmd3, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Paper extraction failed: {result.stderr}")
            return False, f"Paper extraction failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Papers extracted successfully!")

        # Step 4: Assess relevance of extracted papers
        logger.info("Step 4: Assessing relevance of extracted papers...")
        cmd4 = ["python", "-m", "contribution.relevance_assessor", 
                "--research_plan", str(input_file), 
                "--papers_file", str(output_dir / "contribution_papers.json"), 
                "--llm_engine", llm_engine_name,
                "--litellm_name", litellm_name,
                "--output_file", str(output_dir / "filtered_contribution_papers.json"),
                "--cost_log_file", str(contribution_cost_log)]
        result = subprocess.run(cmd4, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Relevance assessment failed: {result.stderr}")
            return False, f"Relevance assessment failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Paper relevance assessed successfully!")

        # Step 5: Augment papers with references and recommendations
        logger.info("Step 5: Augmenting papers with references and recommendations...")
        cmd5 = ["python", "-m", "contribution.paper_augmentation", 
                "--relevant_papers", str(output_dir / "filtered_contribution_papers.json"), 
                "--output_file", str(output_dir / "augmented_contribution_papers.json")]
        result = subprocess.run(cmd5, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Paper augmentation failed: {result.stderr}")
            return False, f"Paper augmentation failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Papers augmented successfully!")

        # Step 6: Embedding filter step 
        logger.info("Step 6: Embedding-based filtering of augmented papers...")
        cmd6 = ["python", "-m", "contribution.embedding_filter", 
                "--research_plan", str(input_file), 
                "--papers_json", str(output_dir / "augmented_contribution_papers.json"), 
                "--output", str(output_dir / "filtered_augmented_contribution_papers.json"), 
                "--top_k", "100"]
        result = subprocess.run(cmd6, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Embedding filter failed: {result.stderr}")
            return False, f"Embedding filter failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Embedding filter successful!")

        # Step 7: Re-assess relevance of augmented papers
        logger.info("Step 7: Re-assessing relevance of augmented papers...")
        cmd7 = ["python", "-m", "contribution.relevance_assessor", 
                "--research_plan", str(input_file), 
                "--papers_file", str(output_dir / "filtered_augmented_contribution_papers.json"), 
                "--llm_engine", llm_engine_name,
                "--litellm_name", litellm_name,
                "--output_file", str(output_dir / "final_contribution_papers.json"),
                "--cost_log_file", str(contribution_cost_log)]
        result = subprocess.run(cmd7, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Final relevance assessment failed: {result.stderr}")
            return False, f"Final relevance assessment failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Final paper set determined!")

        # Step 8: Downsample papers for pairwise comparison
        logger.info("Step 8: Sampling papers for final comparison...")
        cmd8 = ["python", "-m", "contribution.paper_sampler", 
                "--input_file", str(output_dir / "final_contribution_papers.json"), 
                "--output_file", str(output_dir / "sampled_final_contribution_papers.json")]
        result = subprocess.run(cmd8, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Downsampling failed: {result.stderr}")
            return False, f"Downsampling failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Papers sampled successfully!")

        # Step 9: Conduct pairwise comparison
        logger.info("Step 9: Conducting pairwise comparisons...")
        cmd9 = ["python", "-m", "contribution.pairwise_comparator", 
                "--research_plan", str(input_file), 
                "--papers_metadata", str(output_dir / "sampled_final_contribution_papers.json"), 
                "--dimensions_file", str(output_dir / "dimensions_contributions.jsonl"), 
                "--llm_engine", llm_engine_name,
                "--litellm_name", litellm_name,
                "--output_file", str(output_dir / "pairwise_comparisons.json"),
                "--cost_log_file", str(contribution_cost_log)]
        result = subprocess.run(cmd9, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Pairwise comparison failed: {result.stderr}")
            return False, f"Pairwise comparison failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Pairwise comparisons completed!")

        # Step 10: Prepare final contribution context
        logger.info("Step 10: Finalizing comparisons...")
        cmd10 = ["python", "-m", "contribution.prepare_final_contribution_context", 
                 "--input_file", str(output_dir / "pairwise_comparisons.json"), 
                 "--output_file", str(output_dir / "contribution_context.json")]
        result = subprocess.run(cmd10, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Context preparation failed: {result.stderr}")
            return False, f"Context preparation failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Context preparation completed!")

        # Step 11: Generate final contribution review
        logger.info("Step 11: Synthesizing final contribution review...")
        review_file = output_dir / "contribution_review.txt"
        cmd11 = ["python", "-m", "contribution.contribution_review_synthesis", 
                 "--research_plan", str(input_file), 
                 "--comparisons_file", str(output_dir / "contribution_context.json"), 
                 "--llm_engine", llm_engine_name,
                 "--litellm_name", litellm_name,
                 "--output_file", str(review_file),
                 "--cost_log_file", str(contribution_cost_log)]
        result = subprocess.run(cmd11, capture_output=True, text=True, cwd=working_dir, env=env)
        if result.returncode != 0:
            logger.error(f"Contribution review synthesis failed: {result.stderr}")
            return False, f"Contribution review synthesis failed: {result.stderr}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        logger.info("Contribution review completed!")

        # Collect cost information
        cost_info = {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        
        if contribution_cost_log.exists():
            try:
                with open(contribution_cost_log, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            step_data = json.loads(line)
                            cost_info['total_cost'] += step_data.get('cost', 0)
                            cost_info['total_input_tokens'] += step_data.get('input_tokens', 0)
                            cost_info['total_output_tokens'] += step_data.get('output_tokens', 0)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not read cost log {contribution_cost_log}: {e}")
        
        logger.info(f"Contribution workflow total cost: ${cost_info['total_cost']:.4f}")

        # Read the final output
        if review_file.exists():
            with open(review_file, 'r') as f:
                contribution_output = f.read()
            logger.info("Contribution workflow completed successfully!")
            return True, contribution_output, cost_info
        else:
            logger.error("Contribution review output file not found!")
            return False, "Contribution review output file not found!", cost_info

    except Exception as e:
        logger.error(f"Contribution workflow failed: {e}")
        return False, f"Contribution workflow failed: {e}", {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}

def main():
    parser = argparse.ArgumentParser(description='Simple ScholarEval Pipeline for evaluating research plans')
    parser.add_argument('--research_idea', required=True, type=Path,
                       help='Path to the research plan text file')
    parser.add_argument('--llm_engine_name', required=True,
                       help='LLM engine name for processing')
    parser.add_argument('--litellm_name', default='claude-sonnet-4-20250514',
                       help='LiteLLM model name for cost calculation')
    
    args = parser.parse_args()
    
    # Validate research plan file
    if not args.research_idea.exists():
        print(f"Error: Research plan file {args.research_plan} does not exist")
        return 1
    
    # Create output directory
    run_name = f"run_{int(time.time())}"
    output_dir = Path("runs") / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(output_dir)
    logger.info(f"Starting ScholarEval pipeline for: {args.research_plan}")
    logger.info(f"Output directory: {output_dir}")
    
    # Copy research plan to output directory
    research_plan_copy = output_dir / "research_plan.txt"
    with open(args.research_idea, 'r') as src:
        plan_content = src.read()
    with open(research_plan_copy, 'w') as dst:
        dst.write(plan_content)
    
    start_time = time.time()
    
    # Create subdirectories for workflows
    soundness_dir = output_dir / "soundness"
    contribution_dir = output_dir / "contribution"
    soundness_dir.mkdir(exist_ok=True)
    contribution_dir.mkdir(exist_ok=True)
    
    # Copy research plan to workflow directories
    soundness_input = soundness_dir / "research_plan.txt"
    contribution_input = contribution_dir / "research_plan.txt"
    
    with open(soundness_input, 'w') as f:
        f.write(plan_content)
    with open(contribution_input, 'w') as f:
        f.write(plan_content)

    # Run both workflows in parallel
    logger.info("Running soundness and contribution workflows in parallel...")
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both workflows
        soundness_future = executor.submit(
            run_soundness_workflow, 
            soundness_input, soundness_dir, args.llm_engine_name, args.litellm_name, logger
        )
        contribution_future = executor.submit(
            run_contribution_workflow, 
            contribution_input, contribution_dir, args.llm_engine_name, args.litellm_name, logger
        )
    
        # Wait for both to complete
        try:
            soundness_success, soundness_output, soundness_cost_info = soundness_future.result()
        except Exception as e:
            logger.error(f"Soundness workflow failed with exception: {e}")
            soundness_success = False
            soundness_output = f"Soundness workflow failed with exception: {e}"
            soundness_cost_info = {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}
        
        try:
            contribution_success, contribution_output, contribution_cost_info = contribution_future.result()
        except Exception as e:
            logger.error(f"Contribution workflow failed with exception: {e}")
            contribution_success = False
            contribution_output = f"Contribution workflow failed with exception: {e}"
            contribution_cost_info = {'total_cost': 0, 'total_input_tokens': 0, 'total_output_tokens': 0}

    # Calculate total cost
    total_cost = soundness_cost_info['total_cost'] + contribution_cost_info['total_cost']
    total_input_tokens = soundness_cost_info['total_input_tokens'] + contribution_cost_info['total_input_tokens']
    total_output_tokens = soundness_cost_info['total_output_tokens'] + contribution_cost_info['total_output_tokens']

    # Save outputs to main output directory
    soundness_file = output_dir / "soundness.txt"
    contribution_file = output_dir / "contribution.txt"

    if soundness_success:
        with open(soundness_file, 'w') as f:
            f.write(soundness_output)
        logger.info(f"Soundness review saved to: {soundness_file}")
    else:
        with open(soundness_file, 'w') as f:
            f.write(f"Soundness workflow failed: {soundness_output}")
        logger.error("Soundness workflow failed")

    if contribution_success:
        with open(contribution_file, 'w') as f:
            f.write(contribution_output)
        logger.info(f"Contribution review saved to: {contribution_file}")
    else:
        with open(contribution_file, 'w') as f:
            f.write(f"Contribution workflow failed: {contribution_output}")
        logger.error("Contribution workflow failed")

    end_time = time.time()
    total_time = end_time - start_time

    # Create summary report
    summary = {
        "research_plan": str(args.research_plan),
        "run_name": run_name,
        "output_directory": str(output_dir),
        "soundness_success": soundness_success,
        "contribution_success": contribution_success,
        "processing_time_minutes": total_time / 60,
        "total_cost": total_cost,
        "soundness_cost": soundness_cost_info['total_cost'],
        "contribution_cost": contribution_cost_info['total_cost'],
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "soundness_file": str(soundness_file) if soundness_success else None,
        "contribution_file": str(contribution_file) if contribution_success else None
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print final results
    logger.info("=" * 60)
    logger.info("SCHOLAREVAL PIPELINE COMPLETED")
    logger.info("=" * 60)
    logger.info(f"Run: {run_name}")
    logger.info(f"Processing time: {total_time/60:.1f} minutes")
    logger.info(f"Soundness: {'SUCCESS' if soundness_success else 'FAILED'}")
    logger.info(f"Contribution: {'SUCCESS' if contribution_success else 'FAILED'}")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)

    print(f"\nScholarEval pipeline completed!")
    print(f"Run: {run_name}")
    print(f"Results saved to: {output_dir}")
    print(f"Total cost: ${total_cost:.4f}")
    
    if soundness_success and contribution_success:
        return 0
    elif soundness_success or contribution_success:
        return 1
    else:
        return 2

if __name__ == "__main__":
    sys.exit(main())
