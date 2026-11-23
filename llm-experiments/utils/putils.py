import os
import csv

from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd 

def read_functional_groups(config):
    functional_groups = []
    with open(config.probing.fgroups,'r') as f:
        functional_groups = [line.strip() for line in f]
        
    return functional_groups

def read_single_fgroup_train(config, fgroup):
    # Read training data for a single functional group.
    data = {}
    with open(os.path.join(config.probing.data_path, f"undersampled_data_files/seed_{config.probing.probing_seed}", f"{fgroup}_train.csv"),'r') as f:
        csv_reader = csv.reader(f, delimiter=",")
        header = csv_reader.__next__()
        y_index = header.index(f"{fgroup}")
        smiles_idx = header.index("smiles_canon")
        idx = header.index("orig_idx")
        for row in csv_reader:
            data[row[idx]] = {"y_true":int(row[y_index]),
                            "smiles_canon":str(row[smiles_idx])}
        return data

def read_single_fgroup_test(config, fgroup):
    # Read training data for a single functional group.
    data = {}
    with open(os.path.join(config.probing.data_path, f"PCQM4Mv2_2d_3d_test_seed_{config.probing.probing_seed}.csv"),'r') as f:
        csv_reader = csv.reader(f, delimiter=",")
        header = csv_reader.__next__()
        y_index = header.index(f"{fgroup}")
        smiles_idx = header.index("smiles_canon")
        idx = header.index("orig_idx")
        for row in csv_reader:
            data[idx] = {"y_true":int(row[y_index]),
                            "smiles_canon":str(row[smiles_idx])}
        return data

def read_general_data(config):
    # Read training data for a single functional group.
    data = pd.read_csv(os.path.join(config.probing.data_path, f"data_files/PCQM4Mv2_2d_3d_train_seed_{config.probing.probing_seed}.csv"), sep=',')

    return data

def compute_metrics(prediction, answers, print_result=True):
    fpr, tpr, _ = roc_curve(np.array(answers, dtype=bool), -np.array(prediction))
    rauc = auc(fpr, tpr)

    #tpr_5_fpr = tpr[np.where(fpr < 0.05)[0][-1]]

    return fpr, tpr, rauc #, tpr_5_fpr

def evaluate(probe, test_acts, test_data):
    scores = probe.score(test_acts)

    predictions = []
    labels = []
    for i, ex in tqdm(enumerate(test_data)):
        predictions.append(-scores[i].item())
        labels.append(ex["label"])

    fpr, tpr, auc, tpr_5_fpr = compute_metrics(predictions, labels, print_result=False)
    return auc

    
def collect_acts(
    self, dataset_name, model_name, layer, center=True, scale=True, device="cpu"
):
    directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "acts",
        model_name,
        dataset_name,
    )
    activation_files = glob(os.path.join(directory, f"layer_{layer}_*.pt"))
    acts = [
        torch.load(os.path.join(directory, f"layer_{layer}_{i}.pt")).to(device)
        for i in range(0, 25 * len(activation_files), 25)
    ]
    acts = torch.cat(acts, dim=0).to(device)
    if center:
        acts = acts - torch.mean(acts, dim=0)
    if scale:
        acts = acts / torch.std(acts, dim=0)
    return acts

def get(self, layer):
    return self.data[layer]


