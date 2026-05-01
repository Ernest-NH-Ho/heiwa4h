import torch
import traceback
from modules.config import DataConfig, random_seed
from modules.esm3_script.esm3_generation import generate_proteins
import argparse


def main(data_config, use_heiwa):
    try:
        # run_dssp_parallel(data_config=data_config)
        if data_config.MODE == "cleavage":
            print("cleavage mode activated")
            DF = generate_proteins(data_config=data_config, use_heiwa=use_heiwa)
        elif data_config.MODE == "naive":
            print('naive mode activated')
            generate_proteins(data_config=data_config, use_heiwa=use_heiwa)
    except Exception as e:
        with open(f"error.txt", "a+") as out:
            out.write(f"{e}\n")
            traceback.print_exc(file=out)  # 
            out.write(f"\n***************\n")
    
    
if __name__ == '__main__':
    torch.cuda.empty_cache()
    
    # Create the argument parser
    parser = argparse.ArgumentParser(description="A script that takes an input value.")
    parser.add_argument("-heiwa", type=int, required=True, help="0 or 1")  # Add an argument for input
    args = parser.parse_args()
    use_heiwa = bool(args.heiwa)
    
    with torch.no_grad():
        data_config = DataConfig()
        assert data_config.MODE in ["naive", "cleavage"]
        torch.manual_seed(random_seed)
        main(data_config=data_config, use_heiwa=use_heiwa)
    
