import logging 

from . import (lipo, esol)

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

# Returns Task instance of a specified dataset
def get_task(config):
    logging.info(f"Selected {config.task}")
    if config.task == "lipo":
        return lipo.Lipophilicity(config)
    elif config.task == "esol":
        return esol.Esol(config)
    else:
        logging.info("Available tasks:")
        logging.info(TASK_REGISTRY)
        raise KeyError(f"Missing task {config.task}")
