Here’s a cleaned-up version of your README with a clearer structure and a **“Cite Our Work”** section at the end:

````markdown
# ScholarEval: Research Idea Evaluation Grounded in Literature 

## Overview  
<img width="1047" height="482" alt="image" src="https://github.com/user-attachments/assets/9cdff5da-cd83-430e-ae99-a8aa496caec2" />

This repository hosts **ScholarEval**, a retrieval-augmented framework for evaluating research ideas along two key dimensions:  
- **Soundness** — the empirical validity of proposed methods based on existing literature  
- **Contribution** — the degree of advancement relative to prior work  

To support evaluation, we also release **ScholarIdeas**, the first expert-annotated dataset of 117 research ideas and reviews spanning four domains: artificial intelligence, neuroscience, biochemistry, and ecology.  

### Key Results
Our experiments show that ScholarEval:  
- Achieves significantly higher coverage of expert-annotated rubrics in ScholarIdeas compared to all baselines  
- Is consistently preferred over OpenAI’s *o4-mini-deep-research* in terms of actionability, depth, and evidence support  
- Outperforms deep research in a user study on literature engagement, idea refinement, and overall usefulness  

---

## Repository Structure  

### ScholarEval Pipeline  
<img width="1054" height="633" alt="image" src="https://github.com/user-attachments/assets/21e3cd3a-cede-4259-88f8-9a6fab0d2cea" />

The pipeline to generate **soundness** and **contribution** evaluations for a research idea using ScholarEval is located in `ScholarEval/scholareval.py`.  

To run the pipeline:  

```bash
cd ScholarEval
./run_scholareval.sh <research_plan_path> <llm_engine_name> [litellm_name]
````

#### Command Line Arguments

* `research_plan_path`: Path to the research plan text file
* `llm_engine_name`: LLM engine name (e.g., `gpt-4o`, `claude-sonnet-4`)
* `litellm_name`: LiteLLM model name for cost calculation

---

### ScholarIdeas Dataset

<img width="1067" height="373" alt="image" src="https://github.com/user-attachments/assets/68584122-3ff6-46ef-bb3a-3e8f8d38592b" />

The `ScholarIdeas/` directory contains 117 research ideas and their reviews spanning AI, neuroscience, biochemistry, and ecology. Each review is composed of multiple rubrics, for a total of **1076 rubrics** across the dataset.

Each folder in `ScholarIdeas/` includes:

* `{discipline}_{number}.txt` — the research idea text
* `{discipline}_{number}.jsonl` — review rubrics (one rubric per line)
* `cutoff.txt` — cutoff date for literature search (i.e., publication date of the original paper)

Additional resources:

* Dataset creation pipeline: `dataset_creation/`
* Evaluation pipeline (coverage, reference invalidity, LLM metrics): `evaluation/`

---

## Cite Our Work

If you use **ScholarEval** or **ScholarIdeas** in your research, please cite our paper:

```bibtex
@inproceedings{moussa2025scholareval,
  title     = {ScholarEval: Research Idea Evaluation Grounded in Literature},
  author    = {Moussa, Hanane Nour and Queiroz Da Silva, Patrick and Majumder, Bodhisattwa Prasad and Kumar, Sachin and others},
  booktitle = {Proceedings of the Conference on Language Modeling (COLM)},
  year      = {2025}
}
```

