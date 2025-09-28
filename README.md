# ScholarEval: Research Idea Evaluation Grounded in Literature 

## Overview  
<img width="1047" height="482" alt="image" src="https://github.com/user-attachments/assets/9cdff5da-cd83-430e-ae99-a8aa496caec2" />


This repo hosts **ScholarEval**, a retrieval-augmented framework for evaluating research ideas along two key dimensions:  
- **Soundness** — the empirical validity of proposed methods based on existing literature  
- **Contribution** — the degree of advancement relative to prior work  

To evaluate ScholarEval, we also release **ScholarIdeas**, the first expert-annotated dataset of 117 research ideas and reviews spanning four domains: artificial intelligence, neuroscience, biochemistry, and ecology.  

Our experiments show that ScholarEval:  
- Achieves significantly higher coverage of expert-annotated rubrics in ScholarIdeas compared to all baselines  
- Is consistently preferred over OpenAI’s *o4-mini-deep-research* in terms of actionability, depth, and evidence support  
- Based on a user study with experts, outperforms deep research on literature engagement, idea refinement, and overall usefulness  


## In this repository

### ScholarEval Pipeline
<img width="1054" height="633" alt="image" src="https://github.com/user-attachments/assets/21e3cd3a-cede-4259-88f8-9a6fab0d2cea" />

The pipeline to generate soundness and contribution evaluations for a research idea using ScholarEval is located in `ScholarEval/scholareval.py`. To run the pipeline, execute: 

```bash
cd ScholarEval
./run_scholareval.sh <research_plan_path> <llm_engine_name> [litellm_name]
```

#### Command Line Arguments

- `research_plan_path`: Path to the research plan text file
- `llm_engine_name`: LLM engine name (e.g., 'gpt-4o', 'claude-sonnet-4')
- `litellm_name`: LiteLLM model name for cost calculation

### ScholarIdeas Dataset

`ScholarIdeas/` contains 117 research ideas and their reviews spanning artificial intelligence, neuroscience, biochemistry, and ecology. Each review is composed of multiple review rubrics, for a total of 1076 rubrics across the dataset. 
<img width="1067" height="373" alt="image" src="https://github.com/user-attachments/assets/68584122-3ff6-46ef-bb3a-3e8f8d38592b" />

Each folder `ScholarIdeas/` contains:
- `{discipline}_{number}.txt`: The research idea unique identifier
- `{discipline}_{number}.jsonl`: Review rubrics, each line represents a separate rubric. 
- `cutoff.txt`: Cutoff date for literature search (i.e. publication date of the original paper from which the research idea is extracted)

Additionally, the pipeline for ScholarIdeas creation can be found in `dataset_creation/`. 

The evaluation pipeline (coverage, reference invalidity, and LLM metrics) can be found in `evaluation/`. 
