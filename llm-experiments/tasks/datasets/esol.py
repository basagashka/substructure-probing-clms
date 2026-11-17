import os
import csv
import json
import logging 
from itertools import compress

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

from tasks.base import Task
import utils

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class Esol(Task):
    """A task represents an entire benchmark including its dataset, problems,
    answers, generation settings and evaluation methods.
    """
    def __init__(self, config):
        self.dataset_path = os.path.join(config.data_path, config.task)
        self.response_file = config.eval.response_file
        self.get_fgroups = config.inference.get_fgroups
        
    def get_dataset(self, split="test"):
        # Loader for dataset
        logging.info(f"Loading : {self.dataset_path}")

        data = {}
        with open(os.path.join(self.dataset_path, f"{split}.csv"),'r') as f:
            csv_reader = csv.reader(f, delimiter=",")
            header = csv_reader.__next__()
            y_index = header.index("measured log solubility in mols per litre")
            smiles_idx = header.index("smiles_canon")
            #if get_preds:
            #    y_molformer_index = header.index("y_pred_molformer")
            #    y_robertaz_index = header.index("y_pred_roberta_zinc_480m")
            if self.get_fgroups:
                x_start_index = header.index("NHOHCount")
                x_end_index = header.index("fr_urea")
                fgroups = header[x_start_index:x_end_index+1]
            
            for row in csv_reader:
                data[row[0]] = {"y_true":float(row[y_index]),
                                "smiles_canon":str(row[smiles_idx])}
                #if get_preds:
                #    data[row[0]]["y_molformer"] = float(row[y_molformer_index])
                #    data[row[0]]["y_robertaz"] = float(row[y_robertaz_index])
                if self.get_fgroups:
                    # Only check if feature is present or not
                    features = [bool(x) for x in row[x_start_index:x_end_index+1]]
                    data[row[0]]["fgroups"]= list(compress(fgroups, features))
                    
        if self.get_fgroups:
            return data, fgroups
        else:
            return data
        
    def extract_response(self, generation):
        # Extract the respons from a generated text
        pred, floats = utils.fetch_prediction(generation)
        return pred, floats 

    def load_responses(self):
        with open(self.response_file, 'r') as json_file:
            responses = list(json_file)
        self.y_true = []
        self.y_pred = []
        self.invalid = []
        for r in responses:
            instance = json.loads(r)
            pred, floats = utils.fetch_prediction(instance["response"])
            true = float(instance["y_true"])
            if pred is None:
                self.invalid.append((instance["index"], floats, instance["response"]))
                continue
                
            self.y_true.append(true)
            self.y_pred.append(pred)
            
        logging.info(f"Processed results with {len(self.invalid)} invalid responses.")     

    def process_results(self):
        self.load_responses()
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = root_mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        pearson = pearsonr(self.y_true, self.y_pred)
        logging.info("Evaluation finished.")
        results = {'mse': mse, 'rmse':rmse, 'mae':mae, 'pearson': pearson}
        return results
        








