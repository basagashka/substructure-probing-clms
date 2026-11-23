# LLM Experiments

This is base code to run the dwonstream tasks using LLMs

## Installation

To create an environment, you can use conda via: 

    conda create --name=llm-experiments python=3.13.5
    conda activate llm-experiments
    pip install -r requirements.txt 


## Evaluation 

Evaluation can be run via:

    python __main__.py mode="eval"  eval.response_file="results/gpt_oss/4096_low_basic_responses.jsonl" eval.summary="results/gpt_oss/4096_low_esol_basic_summary.jsonl"  

## Inference

You can run inference via:

    python __main__.py mode="inference" task="esol" inference.effort="high" model="/scratch/common_models/gpt-oss-20b" prompts="data/prompts/esol/basic.json"  results_path="results/gpt_oss" inference.get_fgroups=false inference.get_hints=false inference.max_new_tokens=4096 

These are the possible flags for running inference:

- effort: _low_, _medium_, _high_. Controls the reasoning effort for gpt-oss models.
- get_fgroups: _true_, _false_. List all molecular substructures that are present in the molecule..
- get_hints: _true_, _false_. Provide a description on which subgroups increase or decrease lipophilicity (or solubility respectively) in general. Note, that this does not provide hints on which subgroups are present in a molecule. 