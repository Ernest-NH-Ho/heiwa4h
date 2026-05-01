import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

def run_dssp(in_path, out_path):
    if not os.path.exists(out_path):
        subprocess.run(["mkdssp", 
                        "-i", in_path,  
                        "-o", out_path])

def run_dssp_parallel(data_config):
    with ThreadPoolExecutor(max_workers=data_config.NUM_CPU) as executor:
        futures = [executor.submit(run_dssp, 
                                   in_path=f"{data_config.PDB_DIR}/{file}", 
                                   out_path=f"{data_config.DSSP_DIR}/{file.replace('.pdb', '.dssp')}") for file in os.listdir(f"{data_config.PDB_DIR}") if file.endswith('.pdb')]
        for fut in futures:
            fut.result()