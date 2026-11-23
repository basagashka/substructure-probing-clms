import json
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from tasks import datasets

def run_eval(conf):
    """ Given a response file, run the evaluation (creates summary)
    """
    logging.info("Running evaluation...")
    
    data_instance = datasets.get_task(conf)
    results = data_instance.process_results()
    
    with open(conf.eval.summary, 'w') as f:
        json.dump(results, f)
