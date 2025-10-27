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

#### How to run ScholarEval

There are three ways to run ScholarEval on your ideas: running the ScholarEval app locally, running the pipeline via command line, or requesting access to our web app. For any questions or concerns please contact `moussa.45@osu.edu` and `dasilva.30@osu.edu`
##### Run the ScholarEval app locally 
1. Clone this repository.
2. Set the following environment variables
```
export API_KEY="your-litellm-api-key"
export API_ENDPOINT="your-litellm-api-endpoint"
export S2_API_KEY="your-semantic-scholar-api-key"
export email="optional - your academic email for unpaywall open access pdfs"
```
3. Install required packages (see `requirements.txt`)
4. Run `streamlit run ScholarEval/ScholarEval_app.py`
5. Input your research idea and start the soundness and contribution evaluation as shown in [this short demo video](https://youtu.be/rgLWZkdvUPc). All intermediate logs are saved locally in `demo_data.`
##### Run ScholarEval pipeline via command line
Alternatively, if you don't want to use ScholarEval via the user interface, you can simply run the ScholarEval pipeline as follows: 

```bash
bash run_scholareval.sh
````
###### Command Line Arguments
The command line args to set in `run_scholareval.sh` are:
* `research_idea_path`: Path to the research idea text file
* `cutoff_date`: Optional cutoff date for literature search
* `llm_engine_name`: LLM engine name (e.g., `gpt-4o`, `claude-sonnet-4`) as specified by your litellm gateway 
* `save_to`: folder to save all intermediate and final results to
* `litellm_name`: (Optional) LiteLLM model name for cost calculation (e.g., `claude-sonnet-4-5-20250929`)

##### Request access to our web app
And lastly, you can request access to our web app via `go.osu.edu/scholar-eval` We are offering $15 free credits for academic use. 


### ScholarIdeas Dataset

<img width="1290" height="456" alt="image" src="https://github.com/user-attachments/assets/813b041e-acb8-477b-b484-d6ad66bc26c3" />

You can also download ScholarIdeas on [Hugging Face](https://huggingface.co/datasets/hananour/ScholarIdeas). 

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
@misc{moussa2025scholarevalresearchideaevaluation,
      title={ScholarEval: Research Idea Evaluation Grounded in Literature}, 
      author={Hanane Nour Moussa and Patrick Queiroz Da Silva and Daniel Adu-Ampratwum and Alyson East and Zitong Lu and Nikki Puccetti and Mingyi Xue and Huan Sun and Bodhisattwa Prasad Majumder and Sachin Kumar},
      year={2025},
      eprint={2510.16234},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2510.16234}, 
}
```

