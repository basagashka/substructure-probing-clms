import re
import json
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        
def find_floats(input_string):
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    regex = re.compile(numeric_const_pattern, re.VERBOSE)
    floats = regex.findall(input_string)
    return floats
    
def fetch_prediction(input_string):
    floats = find_floats(input_string)
    for i in reversed(floats):
        try: 
            pred = float(i)
            return pred, floats
        except ValueError:
            continue
    logging.info(f"No float found in {input_string}! Returning None.")
    return None, floats
    
def load_fgroup_hints(fgroup_file):
    # Load lists of mappings of hints
    hints = []
    hints_f = json.load(open(fgroup_file, 'r'))
    for _, hint in hints_f.items():
        hints.extend(hint)
    return hints

def load_hints(fgroup_map, fgroup_hints):
    # given a fgroup mapping file and a hint file, load the actual names of the fgroups as hints 
    hints = load_fgroup_hints(fgroup_hints)
    fmap = fetch_fgroup_map(fgroup_map)
    return [fmap[x] for x in hints]

def fetch_fgroup_map(fgroup_file):
    # load simple json file containing the mapping of feature name to the actual name.
    return json.load(open(fgroup_file, 'r'))

def gen_chat_template(model, tokenizer, message, effort, inference_config):

    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, reasoning_effort=effort)
    input = tokenizer(prompt, return_tensors='pt')

    output_tokens = model.generate(
        **input.to(device),
        max_new_tokens = inference_config.max_new_tokens,
        num_return_sequences=1,
        do_sample = True, 
        top_p = inference_config.top_p,
        temperature = inference_config.temperature,
    )

    #generated = output_tokens[0][input["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(output_tokens[0], clean_up_tokenization_spaces=True, skip_special_tokens=True)

    return generated_text

def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

    return tokenizer, model


