import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import model_cost
import threading
import time

from ..engine.litellm_engine import LLMEngine
from ..utils.string_utils import StringUtils


def extract_review_sentences(review_text: str, research_plan_text: str, llm: LLMEngine, llm_cost) -> tuple:
    """
    Process review and research plan pair to generate intermediate output.
    Placeholder function that will be modified later.
    """
    prompt_text = """You are an expert assistant skilled at extracting key components of scientific paper reviews containing evaluations of the underlying reseach idea and discarding all other criticism of other aspects of the scientific paper including results and presentation. Namely, you will be given a text containing all the reviews that the scientific paper received as well as a research plan detailing the main research idea contained in the scientific paper, and your job is to extract sentences from the review that address components mentioned in the research plan. Any sentences that contain any criticism of aspects of the paper not mentioned in the research plan should not be extracted. This is an important guideline to follow. 

    # Reviews contain the following components:
    - Summary: A brief overview of the paper's contributions, main ideas, and objectives.
    - Strengths: Positive aspects highlighted by the reviewer, such as innovative approaches, solid methodology, or valuable contributions.
    - Weaknesses: Negative aspects or concerns raised by the reviewer, such as lack of clarity, methodological flaws, or unanswered questions.
    - Questions: Queries or points of clarification raised by the reviewer, often regarding experimental design, assumptions, or aspects of the research plan.
    
    # The research plan corresponding to the research paper includes:
    - Problem: Background information on the problem the paper addresses, the motivation for pursuing it, and any hypotheses formed by the authors.
    - Method: The methodology proposed by the authors to address the problem.
    - Experiment Design: Details on the experiments, data collection, and evaluation techniques that the authors plan to employ.
    Review:
    {review_text}
    
    Research Plan:
    {research_plan_text}
    
    Output format: 
    This is an extraction task, not a generation task. So your job is exactly to extract sentences from the reviews which address the ideation behing the research (i.e. the research at the idea stage, imagine that no results have been obtained or paper written yet). These would only include criticism related to the problem that the scientists are trying to solve, the methodology they're employing, or the experiments they plan to conduct. Any other content related to typographical errors, presentation or formatting issues, or any results, comments on figures and tables, artifacts, or conclusions derived from executing the experiments should absolutely not be included. Only points related to the underlying research idea as specified in the research plan should be kept. Any point not related to an aspect explicitly mentioned in the research plan should not be included. Return the extracted sentences (verbatim! no changes made) as follows: 
    [Extracted Review: <verbatim extracted sentences>]
    Nothing else besides the extracted review should be contained in the response. 
    """
    
    prompt = [
        {
            "role": "user",
            "content": prompt_text.format(review_text=review_text, research_plan_text=research_plan_text)
        }
    ]
    
    response, prompt_tokens, completion_tokens = llm.respond(prompt, temperature=0.1)
    query_cost = llm_cost["input_cost_per_token"] * prompt_tokens + llm_cost["output_cost_per_token"] * completion_tokens
    return response, query_cost

def polish_review_summary(intermediate_text: str, research_plan_text: str, llm: LLMEngine, llm_cost) -> tuple:
    """
    Process intermediate file and research plan to generate final output.
    Placeholder function that will be modified later.
    """
    prompt_text = """You are an expert assistant skilled at extracting clear and non-redundant evaluation statements from a review summary. In this task you will be given a summary that was created by merely concatenating extracted sentences from a longer review. You will be given this extractive review summary along with the research plan that it reviews and you have to do the following tasks while taking into account the constraints.
    Tasks:
    - Removing redundancies: The review summary was created by concatenating the extracted sentences from multiple reviews written by different reviewers about the same paper. Thus, some of the sentences can be redundant. You should remove all redundancies. If a point is mentioned more the once, drop the other mentions and only keep one.
    - Express each evaluation (weakness or strength mentioned by the reviewer) as a clear statement. Hence your final output should be a list of such statements, each one addressing a specific point made by the reviewer(s).
    - Adding context: the review summary might refer to some aspects of the research plan without sufficient context. You should aim to make each statement in the list clearer by adding context from the research plan that would enhance the understanding of the review.
    - Remove mentions of the final paper: These statements are extracted from the reviews of a paper submitted to a conference or journal. Hence, some statements might have mentions of the final paper (e.g. "in line so and so ..." or "in Figure x ..."). When such statements are encountered, you should slightly rephrase them - without alterations to the meaning - to ensure that they do not refer to the paper but rather to the corresponding component in the research plan. Ensuring this consistency between the statements and the research plan content in crucial. 
    - For each statement (item in the list), you should also annotate it with whether it is a 'strength' or a 'weakness', whether it is related to the paper's 'soundness' (correctness of the methods etc) or 'contribution' (novelty and contribution to the field), and finally annotate whether it represents a 'major' or 'minor' point. Major points are those that reviewers highly stress on and affect the overall evaluation of the paper, while minor points are those that are mentioned but not stressed on as much.

    Constraints:
    - Make sure that the changes that you make are minimal and only aim to remove the redundancy, express the statements as a list, and add context and ensure consistency with the research plan content.
    - Do not completely paraphrase the review summary, try to keep the original wording and sentences as much as possible. Your changes should not lead to a significant overhaul of the review summary.
    - Do not change the tone of the review. It should still be written as if it's produced by an actual reviewer.
    
    Extractive review summary:
    {intermediate_text}
    
    Research Plan:
    {research_plan_text}

    

    Please output a parseable JSONL (JSON Lines) block where each item in the JSONL is a JSON object on a single line as follows:
    ```jsonl
    {{"statement": "<one clear, self-contained statement>", "type": "strength", "axis": "soundness", "severity": "major"}}
    {{"statement": "<another clear, self-contained statement>", "type": "weakness", "axis": "contribution", "severity": "minor"}}
    {{"statement": "<another clear, self-contained statement>", "type": "strength", "axis": "soundness", "severity": "minor"}}
    ```
    
    Note: Each JSON object must be on a single line. The "type" field should be either "strength" or "weakness", the "axis" field should be either "soundness" or "contribution", and the "severity" field should be either "major" or "minor".

    Nothing else besides this JSONL block should be contained in the response.
    
    """
    
    prompt = [
        {
            "role": "user",
            "content": prompt_text.format(intermediate_text=intermediate_text, research_plan_text=research_plan_text)
        }
    ]
    
    response, prompt_tokens, completion_tokens = llm.respond(prompt, temperature=0.1)
    query_cost = llm_cost["input_cost_per_token"] * prompt_tokens + llm_cost["output_cost_per_token"] * completion_tokens
    return response, query_cost

def read_text_file(file_path: str) -> str:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_review_research_pair(review_file: str, reviews_folder: str, research_plans_folder: str, 
                                intermediate_folder: str, llm: LLMEngine, rate_limit_lock: threading.Lock, 
                                last_request_time: List[float], rate_limit_delay: float, llm_cost) -> Dict[str, Any]:
    base_name = os.path.splitext(review_file)[0]
    review_path = os.path.join(reviews_folder, review_file)
    research_plan_path = os.path.join(research_plans_folder, f"{base_name}.txt")
    
    try:
        review_text = read_text_file(review_path)
        if review_text is None:
            return {"error": f"Failed to read review file {review_file}", "file": review_file}
        
        if not os.path.exists(research_plan_path):
            return {"error": f"No matching research plan found for {review_file}", "file": review_file}
        
        research_plan_text = read_text_file(research_plan_path)
        if research_plan_text is None:
            return {"error": f"Failed to read research plan file for {base_name}", "file": review_file}
        
        with rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - last_request_time[0]
            if time_since_last < rate_limit_delay:
                time.sleep(rate_limit_delay - time_since_last)
            last_request_time[0] = time.time()
            
            intermediate_result, query_cost = extract_review_sentences(review_text, research_plan_text, llm, llm_cost)
        
        intermediate_file_path = os.path.join(intermediate_folder, f"{base_name}_intermediate.txt")
        with open(intermediate_file_path, 'w', encoding='utf-8') as f:
            f.write(intermediate_result)
        
        return {
            "success": True,
            "file": review_file,
            "intermediate_file": intermediate_file_path,
            "query_cost": query_cost
        }
        
    except Exception as e:
        return {"error": f"Error processing {review_file}: {e}", "file": review_file}


def process_intermediate_file(intermediate_file: str, intermediate_folder: str, research_plans_folder: str,
                             final_folder: str, llm: LLMEngine, rate_limit_lock: threading.Lock,
                             last_request_time: List[float], rate_limit_delay: float, llm_cost) -> Dict[str, Any]:
    base_name = os.path.splitext(os.path.basename(intermediate_file))[0].replace('_intermediate', '')
    intermediate_path = os.path.join(intermediate_folder, intermediate_file)
    research_plan_path = os.path.join(research_plans_folder, f"{base_name}.txt")
    su = StringUtils()
    
    try:
        intermediate_text = read_text_file(intermediate_path)
        if intermediate_text is None:
            return {"error": f"Failed to read intermediate file {intermediate_file}", "file": intermediate_file}
        
        if not os.path.exists(research_plan_path):
            return {"error": f"No matching research plan found for {base_name}", "file": intermediate_file}
        
        research_plan_text = read_text_file(research_plan_path)
        if research_plan_text is None:
            return {"error": f"Failed to read research plan file for {base_name}", "file": intermediate_file}
        
        with rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - last_request_time[0]
            if time_since_last < rate_limit_delay:
                time.sleep(rate_limit_delay - time_since_last)
            last_request_time[0] = time.time()
            
            final_result, query_cost = polish_review_summary(intermediate_text, research_plan_text, llm, llm_cost)
        
        final_result = su.extract_jsonl_output(final_result)
        
        final_file_path = os.path.join(final_folder, f"{base_name}.jsonl")

        with open(final_file_path, 'w') as f:
            for json_obj in final_result:
                json.dump(json_obj, f)
                f.write('\n')
        
        return {
            "success": True,
            "file": intermediate_file,
            "final_file": final_file_path,
            "query_cost": query_cost
        }
        
    except Exception as e:
        return {"error": f"Error processing {intermediate_file}: {e}", "file": intermediate_file}


def main():
    parser = argparse.ArgumentParser(description="Process reviews and research plans through two-stage analysis")
    parser.add_argument("--reviews_folder", required=True, help="Folder containing review txt files")
    parser.add_argument("--research_plans_folder", required=True, help="Folder containing research plan txt files")
    parser.add_argument("--intermediate_folder", required=True, help="Folder to save intermediate outputs")
    parser.add_argument("--final_folder", required=True, help="Folder to save final outputs")
    parser.add_argument("--llm_engine_name", required=True, help="LLM engine name (e.g., 'gpt-4o')")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of concurrent processors (default: 3)")
    parser.add_argument("--rate_limit_delay", type=float, default=1.0, help="Delay between LLM calls in seconds (default: 1.0)")
    parser.add_argument("--skip_extraction", action="store_true", help="Skip Stage 1 (extraction) and go directly to Stage 2 (polishing) using existing intermediate files")
    args = parser.parse_args()

    os.makedirs(args.intermediate_folder, exist_ok=True)
    os.makedirs(args.final_folder, exist_ok=True)

    rate_limit_lock = threading.Lock()
    last_request_time = [0.0]

    API_KEY = os.environ.get("API_KEY_1")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    llm_cost = model_cost["claude-sonnet-4-20250514"]

    stage1_results = []
    failed_stage1 = []
    total_stage1_cost = 0.0
    
    if not args.skip_extraction:
        review_files = [f for f in os.listdir(args.reviews_folder) if f.endswith('.txt')]
        total_files = len(review_files)
        
        print(f"Found {total_files} review files to process")

        print("\nStage 1: Processing review-research plan pairs...")

        def process_stage1_wrapper(review_file: str) -> Dict[str, Any]:
            return process_review_research_pair(
                review_file, args.reviews_folder, args.research_plans_folder,
                args.intermediate_folder, llm, rate_limit_lock, last_request_time, args.rate_limit_delay, llm_cost
            )

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            future_to_file = {executor.submit(process_stage1_wrapper, review_file): review_file for review_file in review_files}
            
            with tqdm(total=total_files, desc="Stage 1: Review-Research Plan pairs") as pbar:
                for future in as_completed(future_to_file):
                    review_file = future_to_file[future]
                    
                    try:
                        result = future.result()
                        if result.get('success'):
                            stage1_results.append(result)
                            total_stage1_cost += result.get('query_cost', 0.0)
                            print(f"Stage 1: Successfully processed {review_file}")
                        else:
                            failed_stage1.append(review_file)
                            print(f"Stage 1: Failed to process {review_file}: {result.get('error', 'Unknown error')}")
                    except Exception as e:
                        print(f"Stage 1: Exception processing {review_file}: {e}")
                        failed_stage1.append(review_file)
                    
                    pbar.update(1)

        print(f"Stage 1 completed: {len(stage1_results)}/{total_files} successful")
    else:
        print("Skipping Stage 1: Using existing intermediate files")

    print("\nStage 2: Processing intermediate files...")
    intermediate_files = [f for f in os.listdir(args.intermediate_folder) if f.endswith('_intermediate.txt')]
    stage2_results = []
    failed_stage2 = []
    total_stage2_cost = 0.0

    def process_stage2_wrapper(intermediate_file: str) -> Dict[str, Any]:
        return process_intermediate_file(
            intermediate_file, args.intermediate_folder, args.research_plans_folder,
            args.final_folder, llm, rate_limit_lock, last_request_time, args.rate_limit_delay, llm_cost
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {executor.submit(process_stage2_wrapper, intermediate_file): intermediate_file for intermediate_file in intermediate_files}
        
        with tqdm(total=len(intermediate_files), desc="Stage 2: Intermediate files") as pbar:
            for future in as_completed(future_to_file):
                intermediate_file = future_to_file[future]
                
                try:
                    result = future.result()
                    if result.get('success'):
                        stage2_results.append(result)
                        total_stage2_cost += result.get('query_cost', 0.0)
                        print(f"Stage 2: Successfully processed {intermediate_file}")
                    else:
                        failed_stage2.append(intermediate_file)
                        print(f"Stage 2: Failed to process {intermediate_file}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"Stage 2: Exception processing {intermediate_file}: {e}")
                    failed_stage2.append(intermediate_file)
                
                pbar.update(1)

    print(f"Stage 2 completed: {len(stage2_results)}/{len(intermediate_files)} successful")

    total_cost = total_stage1_cost + total_stage2_cost

    summary = {
        "total_review_files": total_files if not args.skip_extraction else "N/A (Stage 1 skipped)",
        "stage1_successful": len(stage1_results),
        "stage1_failed": failed_stage1,
        "stage1_cost": total_stage1_cost,
        "stage2_successful": len(stage2_results),
        "stage2_failed": failed_stage2,
        "stage2_cost": total_stage2_cost,
        "total_cost": total_cost,
        "llm_engine_used": args.llm_engine_name,
        "max_workers": args.max_workers,
        "rate_limit_delay": args.rate_limit_delay,
        "extraction_skipped": args.skip_extraction
    }
    
    summary_file = os.path.join(args.final_folder, 'processing_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nProcessing completed!")
    if not args.skip_extraction:
        print(f"Stage 1: {len(stage1_results)}/{total_files} successful")
        print(f"Stage 1 LLM query cost: ${total_stage1_cost:.6f}")
    else:
        print("Stage 1: Skipped")
    print(f"Stage 2: {len(stage2_results)}/{len(intermediate_files)} successful")
    print(f"Stage 2 LLM query cost: ${total_stage2_cost:.6f}")
    print(f"Total LLM query cost: ${total_cost:.6f}")
    print(f"Intermediate files saved to: {args.intermediate_folder}")
    print(f"Final files saved to: {args.final_folder}")
    print(f"Summary saved to: {summary_file}")


if __name__ == '__main__':
    main()