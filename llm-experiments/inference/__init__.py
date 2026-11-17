import os
import sys
import json
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torch
from tqdm import tqdm

import utils
from eval import run_eval
from tasks import datasets

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def run_inference(conf):

    data_instance = datasets.get_task(conf)
    if conf.inference.get_fgroups:
        inputs, fgroups = data_instance.get_dataset(split="test") # use test split for inference
    else:
        inputs = data_instance.get_dataset(split="test") # use test split for inference
    # Load prompts
    prompts = json.load(open(conf.prompts, 'r'))
    tokenizer, model = utils.load_model(conf.model)

    target_data = inputs
    model_responses = []

    logging.info(f"Generating responses...")
    
    prompt_stub = conf.prompts.split('/')[-1].replace(".json","")
    fgroup_map = utils.fetch_fgroup_map(conf.fgroup_mapping)

    fname = f"{conf.results_path}/{conf.inference.max_new_tokens}_{conf.inference.effort}_{prompt_stub}"
    for idx, item in tqdm(inputs.items()):
    
        message = [
            {"role": "system", "content": prompts["system"]}]
        
        user_prompt = {"role": "user", "content": prompts["user"] }

        if conf.inference.get_hints:
            if conf.task == "esol":
                hint_string = f"For prediction it is also important to consider the following functional groups that affect soluability."
            else:
                hint_string = f"For prediction it is also important to consider the following functional groups that affect lipophilicity."
            lipo_hints = utils.load_hints(conf.fgroup_mapping, conf.lipo_fgroups)
            hydro_hints = utils.load_hints(conf.fgroup_mapping, conf.hydro_fgroups)
            if conf.task == "esol":
                # Hydrophilic structures increase soluability in water
                lipo_prompt = f"These are functional groups that increase soluability:\n{', '.join([x for x in lipo_hints])}\n\n"
                hydro_prompt = f"These are functional groups that decrease soluability:\n{', '.join([x for x in hydro_hints])}\n\n"
            else:
                # Lipophilic structures increase soluability in fats 
                lipo_prompt = f"These are functional groups that increase lipophilicity:\n{', '.join([x for x in lipo_hints])}\n\n"
                hydro_prompt = f"These are functional groups that decrease lipophilicity:\n{', '.join([x for x in hydro_hints])}\n\n"
            hint_string += lipo_prompt
            hint_string += hydro_prompt
            user_prompt["content"] += hint_string  
            
        if conf.inference.get_fgroups:
            # Convert fgroup abbreviations into plain names
            fg_list = ", ".join([fgroup_map[x] for x in item["fgroups"]])
            fg_string = f"As a hint, you are also given the functional groups that are present in the molecule. Following functional groups are present in the molecule: {fg_list}\n\n"
            user_prompt["content"] += fg_string  

        instance_string = f"This is the molecule: {item['smiles_canon']}\n\n"

        user_prompt["content"] += instance_string
        
        message.append(user_prompt)

        response = utils.gen_chat_template(model, tokenizer, message, conf.inference.effort, conf.inference)
        model_responses.append(response)

        pred, floats = utils.fetch_prediction(response) 
                      
        # save the model responses in a file on the fly
        with open(f"{fname}_responses.jsonl", 'a') as f:
            json.dump({'index': idx, 'y_true':item["y_true"], 'y_pred':pred, 'instance': item["smiles_canon"], 'response': response, 'all_pred':floats}, f)
            f.write("\n")

    logging.info("Predictions done.")
    # Run evaluation:
    if conf.inference.include_eval:
        conf.mode = "eval"
        conf.eval.response_file = f"{fname}_responses.jsonl"
        run_eval(conf)


