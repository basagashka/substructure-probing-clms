import os
import sys
import json
import argparse
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import rdkit

from tasks.lipo import Lipophilicity
from tasks.esol import Esol

def write_data(path, data): 
    with open(path, 'w') as f:
        json.dump(data, f)

def main():
    parser = argparse.ArgumentParser(description='Generating fingerprints with RDKit')
    parser.add_argument('--fp',
                        type=str,
                        help='name of the fingerprinting method',
    )
    parser.add_argument('--task', 
                        type=str,
                        help='esol or lipo')

    args = parser.parse_args()

    if args.task == 'lipo':
        dataset = Lipophilicity(f"data/{args.task}/")
    else:
        dataset = Esol(f"data/{args.task}/")
        
    train = dataset.get_dataset("train")
    val = dataset.get_dataset("val")
    test = dataset.get_dataset("test")

    # Efficient lipophilicity prediction of molecules employing deep-learning models
    # uses morgan with radius 1 and dim 300
    # others use radius 2 (equal to ECFP4)

    fp_sizes = [300, 512, 1024, 2048] 

    output_path = f"data/{args.task}/fingerprints/"

    for fp_size in fp_sizes:
        for split in [(train, "train"), (val, "val"), (test, "test")]:
            fp_dat = split[0].copy()
            
            for idx, instance in split[0].items():
                if args.fp == "m":
                    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=fp_size)
                elif args.fp == "rdk":
                    fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
                elif args.fp == "ap":
                    fpgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fp_size)
                elif args.fp == "tt":
                    fpgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fp_size)
                else:
                    logging.info("Please use an existing fingerprint generator")
                    sys.exit()
                
                smiles = Chem.MolFromSmiles(instance["smiles_canon"])
                label = instance["y_true"]
                fp_dat[idx]["fp"]=fpgen.GetFingerprint(smiles).ToList()
            
            write_data(os.path.join(output_path, f"{args.fp}fp_{split[1]}_{fp_size}"), fp_dat)
                        

if __name__ == '__main__':
    main()  
