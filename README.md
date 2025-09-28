## ScholarEval

The pipeline to generate soundness and contribution evaluations for a research idea using ScholarEval is located in `ScholarEval/scholareval.py`.

### Usage

```bash
cd ScholarEval
./run_scholareval.sh <research_plan_path> <llm_engine_name> [litellm_name]
```

### Command Line Arguments

- `research_plan_path`: Path to the research plan text file
- `llm_engine_name`: LLM engine name for processing (e.g., 'gpt-4o', 'claude-sonnet-4')
- `litellm_name`: Optional LiteLLM model name for cost calculation (default: claude-sonnet-4-20250514)

## ScholarIdeas Dataset

The ScholarIdeas dataset is located in `ScholarIdeas/`. It contains research plans organized by discipline:


Each folder contains:
- `{discipline}_{number}.txt`: The research plan
- `{discipline}_{number}.jsonl`: Metadata and structured data
- `cutoff.txt`: Temporal cutoff date for evaluation

## Dataset Creation Process

The dataset creation process is located in `dataset_creation/`:

- `prepare_paper_content.py`: Extracts paper content from PDFs using GROBID (with PyPDF2 fallback), excluding results and conclusion sections
- `extract_research_plan.py`: Extracts research plans from paper content using LLM


## Evaluation Pipeline

The evaluation pipeline is located in `evaluation/`:

- `coverage.py`: Alignment evaluation (renamed from prometheus_alignment_judge.py)
- `reference_invalidity.py`: Reference validity checking with robust heuristics for different HTTP status codes
- `llm_metrics.py`: LLM-based evaluation metrics