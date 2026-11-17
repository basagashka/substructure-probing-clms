import os
import csv
import json
import logging
from itertools import compress

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

class Esol:
    """Esol dataset.
    """
    def __init__(self, datapath):
        self.dataset_path = datapath
      
    def get_fingerprints_dataset(self, fingerprints_file, fingerprints_path = "fingerprints"):
        # Loader for dataset
        data = json.load(open(os.path.join(self.dataset_path, fingerprints_path, fingerprints_file),'r'))
        return data
        
    def get_dataset(self, split="test"):
        # Loader for dataset
        logging.info(f"Loading raw data: {self.dataset_path}")

        fingerprints = False
        data = {}
        with open(os.path.join(self.dataset_path, f"{split}.csv"),'r') as f:
            csv_reader = csv.reader(f, delimiter=",")
            header = csv_reader.__next__()
            y_index = header.index("measured log solubility in mols per litre")
            smiles_idx = header.index("smiles_canon")
            
            for row in csv_reader:
                data[row[0]] = {"y_true":float(row[y_index]),
                                "smiles_canon":str(row[smiles_idx])}
        return data
