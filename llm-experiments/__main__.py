import sys
import random
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import numpy as np
import torch 
import hydra

from eval import run_eval
from inference import run_inference
from probing import run_probe

@hydra.main(version_base=None, config_path="conf", config_name="config")
def experiment(config):
    """ Main experiment script
    """
    #Access the config like this: cfg.parameter_name   
    # Set the seeds for reproducibility
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    torch.backends.cudnn.deterministic = True
    
    if config.mode == "inference":
        logging.info("Running inference")
        run_inference(config)
    elif config.mode == "eval":
        logging.info("Running evaluation")
        run_eval(config)
    else:
        sys.exit("No mode set. Please chose betweeen inference and eval.")


if __name__ == '__main__':
    experiment()
