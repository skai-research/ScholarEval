# ScholarEval

This is the accompanying repository of the paper "ScholarEval: Research Idea Evaluation Grounded in Literature."

## Overview  
<img width="1623" height="695" alt="image" src="https://github.com/user-attachments/assets/8530749b-520b-4dc9-b420-9adb11269526" />


This repository hosts **ScholarEval**, a retrieval-augmented framework for evaluating research ideas along two key dimensions:  
- **Soundness** — the empirical validity of proposed methods based on existing literature  
- **Contribution** — the degree of advancement relative to prior work  

To support evaluation, we also release **ScholarIdeas**, the first expert-annotated dataset of 117 research ideas and reviews spanning four domains: artificial intelligence, neuroscience, biochemistry, and ecology.  

### Key Results
Our evaluation shows that ScholarEval:  
- Achieves significantly higher coverage of expert-annotated rubrics in ScholarIdeas compared to all baselines  
- Is consistently preferred over OpenAI’s *o4-mini-deep-research* in terms of actionability, depth, and evidence support  
- Outperforms deep research in a user study with experts on literature engagement, idea refinement, and overall usefulness  

---

## Repository Structure  

### ScholarEval Pipeline  
<img width="1279" height="757" alt="image" src="https://github.com/user-attachments/assets/a7da367a-3b12-4935-9570-b05afd062536" />

The pipeline to generate **soundness** and **contribution** evaluations for a research idea using ScholarEval is located in `ScholarEval/scholareval.py`.  

To run the pipeline:  

```bash
cd ScholarEval
./run_scholareval.sh <research_plan_path> <llm_engine_name> [litellm_name]
````

#### Command Line Arguments

* `research_idea_path`: Path to the research idea text file
* `llm_engine_name`: LLM engine name (e.g., `gpt-4o`, `claude-sonnet-4`)
* `litellm_name`: LiteLLM model name for cost calculation

---

### ScholarIdeas Dataset

<img width="1290" height="456" alt="image" src="https://github.com/user-attachments/assets/813b041e-acb8-477b-b484-d6ad66bc26c3" />

The `ScholarIdeas/` directory contains 117 research ideas and their reviews spanning AI, neuroscience, biochemistry, and ecology. Each review is composed of multiple rubrics, for a total of **1076 rubrics** across the dataset.

Each folder in `ScholarIdeas/` includes:

* `{discipline}_{number}.txt` — the research idea text
* `{discipline}_{number}.jsonl` — review rubrics (one rubric per line)
* `cutoff.txt` — cutoff date for literature search (i.e., publication date of the original paper)

Additional resources:

* Dataset creation pipeline: `dataset_creation/`
* Evaluation pipeline (coverage, reference invalidity, LLM metrics - actionability, depth, and evidence support): `evaluation/`

---

## Cite Our Work

If you find **ScholarEval** or **ScholarIdeas** useful in your work, please cite our paper:

```bibtex
@misc{moussa2025scholareval,
      title={ScholarEval: Research Idea Evaluation Grounded in Literature}, 
      author={Hanane Nour Moussa, Patrick Queiroz Da Silva, Daniel Adu-Ampratwum, Alyson East, Zitong Lu, Nikki Puccetti, Mingyi Xue, Huan Sun, Bodhisattwa Prasad Majumder, Sachin Kumar},
      year={2025},
      eprint={},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/}, 
}
```

