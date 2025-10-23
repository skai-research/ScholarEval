import os
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from litellm import model_cost
import threading
import time

from ..engine.litellm_engine import LLMEngine


def read_paper_content(content_path: str) -> str:
    try:
        with open(content_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    except Exception as e:
        print(f"Error reading paper content from {content_path}: {e}")
        return None


def process_single_paper_content(content_file: str, paper_content_dir: str, research_plans_dir: str, llm: LLMEngine, 
                                rate_limit_lock: threading.Lock, last_request_time: list, rate_limit_delay: float, llm_cost) -> dict:
    content_path = os.path.join(paper_content_dir, content_file)
    paper_name = os.path.splitext(content_file)[0]
    
    try:
        # Read paper content from file
        paper_content = read_paper_content(content_path)
        
        if paper_content is None:
            return {"error": f"Failed to read paper content from {content_file}", "content_file": content_file}
        
        # Rate-limited LLM call
        with rate_limit_lock:
            current_time = time.time()
            time_since_last = current_time - last_request_time[0]
            if time_since_last < rate_limit_delay:
                time.sleep(rate_limit_delay - time_since_last)
            last_request_time[0] = time.time()
            
            research_plan, query_cost = extract_research_plan_from_content(paper_content, llm, llm_cost)
        
        # Save research plan as txt file
        research_plan_path = os.path.join(research_plans_dir, f"{paper_name}.txt")
        with open(research_plan_path, 'w', encoding='utf-8') as f:
            f.write(research_plan)
        
        return {"success": True, "content_file": content_file, "research_plan_file": research_plan_path, "query_cost": query_cost}
        
    except Exception as e:
        return {"error": f"Error processing {content_file}: {e}", "content_file": content_file}


def extract_research_plan_from_content(paper_content: str, llm: LLMEngine, llm_cost) -> str:
    gen_research_plan_prompt = """You are a highly skilled assistant tasked with thoroughly reading the content of a research paper and extracting its research plan. The research plan should reflect the state of the research at the time it was proposed, before any experiments or conclusions were drawn.

    A research plan consists of the following key sections:

    Problem:

    - This section provides the background of the problem being addressed, the motivation for pursuing it, and any hypotheses that the authors have formed at the start of their research.
    - Focus on the initial framing of the problem, why it's important, and the research questions that guide the study.

    Method:

    - Summarize the methodology that the authors will follow to address the problem. This includes the overall approach, any theoretical frameworks, models, or algorithms to be used, and the rationale behind the chosen methodology.
    - Do not include any references to specific results or findings - only the proposed methods and strategies.

    Experiment Design:

    - Describe the experiments that the authors will conduct to test their hypotheses. This should include the design, variables, procedures, and any tools or techniques that will be employed.
    - Provide details about how the authors plan to gather data, measure outcomes, and assess the effectiveness of the methods, but do not include any results.
    - However, make sure not to make any reference to experimental setups that the authors could not have known before conducting the experiments (e.g. specific values for hyperparameters or other design choices that were made through trial and error).

    Key Constraints:

    - No results or conclusions: Exclude any findings, outcomes, or conclusions that were reached after conducting the experiments. The research plan should represent the research at the proposal stage, not the completion stage.
    - First-person perspective: Write the research plan from the authors' point of view, as if the authors themselves are outlining their proposed work. Avoid referring to the authors in the third person.
    - Concise and focused: The research plan should be clear, concise, and direct. Include only the information necessary to describe the research as it was proposed, not the outcomes.
    - Represent the status before experimentation: Ensure the plan reflects the status of the project when it was still in the idea or proposal phase, prior to conducting any experiments.

    General Writing Guidelines:

    - Use active voice as if it's written by the authors themselves.
    - Maintain a formal yet straightforward tone typical of research proposals.
    - Keep the text brief and to the point, and avoid extraneous details or unnecessary elaboration.

    The research plans should be faithful to the original research proposal, concise, and aligned with the intent of the authors at the time of the project ideation phase.

    Generate the research plan for the following paper:
    {paper_content}"""

    prompt = [
        {
            "role": "user",
            "content": gen_research_plan_prompt.format(paper_content=paper_content)
        }
    ]
    
    response, prompt_tokens, completion_tokens= llm.respond(prompt, temperature=0.1)
    query_cost = llm_cost["input_cost_per_token"] * prompt_tokens + llm_cost["output_cost_per_token"] * completion_tokens
    return response, query_cost


def main():
    parser = argparse.ArgumentParser(description="Extract research plans from paper content files and save as txt files")
    parser.add_argument("--paper_content_dir", required=True, help="Directory containing paper content text files")
    parser.add_argument("--research_plans_dir", required=True, help="Directory to save research plan txt files")
    parser.add_argument("--llm_engine_name", required=True, help="LLM engine name (e.g., 'gpt-4o')")
    parser.add_argument("--max_workers", type=int, default=3, help="Maximum number of concurrent processors (default: 3)")
    parser.add_argument("--rate_limit_delay", type=float, default=1.0, help="Delay between LLM calls in seconds (default: 1.0)")
    args = parser.parse_args()

    os.makedirs(args.research_plans_dir, exist_ok=True)

    rate_limit_lock = threading.Lock()
    last_request_time = [0.0]

    API_KEY = os.environ.get("API_KEY")
    API_ENDPOINT = os.environ.get("API_ENDPOINT")
    llm = LLMEngine(llm_engine_name=args.llm_engine_name, api_key=API_KEY, api_endpoint=API_ENDPOINT)
    llm_cost = model_cost["claude-sonnet-4-20250514"]

    if not os.path.exists(args.paper_content_dir):
        print(f"Error: Paper content directory {args.paper_content_dir} does not exist")
        return
    
    content_files = [f for f in os.listdir(args.paper_content_dir) if f.endswith('.txt')]
    total_files = len(content_files)
    
    if total_files == 0:
        print(f"No paper content files found in {args.paper_content_dir}")
        return
    
    print(f"Found {total_files} paper content files to process in {args.paper_content_dir}")
    
    successful_extractions = []
    failed_extractions = []
    total_cost = 0.0

    def process_content_wrapper(content_file: str) -> dict:
        return process_single_paper_content(content_file, args.paper_content_dir, args.research_plans_dir, llm, rate_limit_lock, last_request_time, args.rate_limit_delay, llm_cost)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_file = {executor.submit(process_content_wrapper, content_file): content_file for content_file in content_files}
        
        with tqdm(total=total_files, desc="Extracting research plans") as pbar:
            for future in as_completed(future_to_file):
                content_file = future_to_file[future]
                
                try:
                    result = future.result()
                    if result.get('success'):
                        successful_extractions.append(content_file)
                        total_cost += result.get('query_cost', 0.0)
                        print(f"Successfully processed {content_file}")
                    else:
                        failed_extractions.append(content_file)
                        print(f"Failed to process {content_file}: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    print(f"Exception processing {content_file}: {e}")
                    failed_extractions.append(content_file)
                
                pbar.update(1)

    print(f"\nExtraction completed!")
    print(f"Successfully processed: {len(successful_extractions)}/{total_files}")
    print(f"Failed extractions: {len(failed_extractions)}")
    print(f"Research plans saved to: {args.research_plans_dir}")
    print(f"Total LLM query cost: ${total_cost:.6f}")
    
    if failed_extractions:
        print(f"Failed files: {failed_extractions}")


if __name__ == '__main__':
    main()