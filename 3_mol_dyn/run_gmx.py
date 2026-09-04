"""
Test script for GromacsRunner
This demonstrates how to use the GromacsRunner class with a sample dataset.
"""
import os
import pandas as pd
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from python_modules.gromacs_runner import GromacsRunner


def setup_test_environment(**kwargs):
    """
    Create a minimal test environment structure.
    This would normally be done by the user.
    """
    print("Setting up test environment...")
    destination_dir = kwargs["destination_dir"]
    # Create directory structure
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(exist_ok=True)
    source_dir = Path(kwargs["source_dir"])
    df = kwargs["client_list"]

    def set_up_protein_dir(row):
        if os.path.exists(source_dir / f"{row['file'].replace('.pdb', '')}.pdb"):
            protein_dir = destination_dir / row["folder"]
            if not os.path.exists(protein_dir / f"{row['file'].replace('.pdb', '')}.pdb"):
                protein_dir.mkdir(exist_ok=True)
                os.rename(source_dir / f"{row['file'].replace('.pdb', '')}.pdb", protein_dir / f"{row['file'].replace('.pdb', '')}.pdb")
        
    with ThreadPoolExecutor(max_workers=kwargs["n_threads"]) as executor:
        futures = [executor.submit(set_up_protein_dir, row) for idx, row in df.iterrows()]
        for fut in futures:
            fut.result()
    
    print("✓ Test directory structure created")
    print()

def move_pdb_back(**kwargs):
    """
    Create a minimal test environment structure.
    This would normally be done by the user.
    """
    print("Setting up test environment...")
    df = kwargs["client_list"]
    destination_dir = kwargs["destination_dir"]
    # Create directory structure
    destination_dir = Path(destination_dir)
    source_dir = Path(kwargs["source_dir"])
    

    def move_pdb(row):
        if os.path.exists(destination_dir / row["folder"] / f"{row['file'].replace('.pdb', '')}.pdb"):
            os.rename(destination_dir / row["folder"] / f"{row['file'].replace('.pdb', '')}.pdb", source_dir / f"{row['file'].replace('.pdb', '')}.pdb")
        
    with ThreadPoolExecutor(max_workers=kwargs["n_threads"]) as executor:
        futures = [executor.submit(move_pdb, row) for idx, row in df.iterrows()]
        for fut in futures:
            fut.result()
    
    print(f"✓ Input pdb returned to source directory {kwargs['source_dir']}")
    print()

def run_full_simulation(task, **kwargs):
    """
    Run the full GROMACS simulation pipeline.
    WARNING: This requires GROMACS to be installed and will take hours to run!
    """
    print("\n" + "="*80)
    print("FULL SIMULATION RUN")
    print("="*80)
    print("\n⚠ WARNING: This will run actual GROMACS simulations!")
    print("This requires:")
    print("  1. GROMACS installation")
    print("  2. Real PDB files in clients/ folders")
    print("  3. GPU availability")
    print("  4. Several hours of computation time\n")
    
    # Load the CSV
    setup_test_environment(**kwargs)
    
    # Initialize runner
    runner = GromacsRunner(task=task, **kwargs)
    
    # Run everything!
    runner.run()

    if kwargs["return_pdb_to_source"]:
        move_pdb_back(**kwargs)


parser = argparse.ArgumentParser(description="A script that takes an input value.")
parser.add_argument("-task", type=str, required=True, help="Acceptable tasks: 'solvate', 'solvate_with_ion', 'em', 'md'")  # Add an argument for input
args = parser.parse_args()
task = args.task

if __name__ == "__main__":
    # Run the dry test (MDP file generation only)
    # test_gromacs_runner_dry_run()
    components = {"client_list": pd.read_csv("file_name.csv",  header=0)
                  "source_dir": "path_to_source", 
                  "destination_dir": "path_to_destination", 
                  "return_pdb_to_source": False, 
                  "run_from_start": False,
                  "box_shape": "dodecahedron", 
                  "box_margin": 1, 
                  "ff": "amber99sbws", # amber19sb, amber99sb-ildn
                  "water_model": "tip4p", "water_coord": "tip4p.gro", # ,  opc
                  "cation":"NA", "anion":"CL", "salt_conc":0.15, 
                  "em_nvt_iter":10, 
                  "dt_em": 0.01, "em_max_steps": 250000, # "dt_em": 0.01, "em_max_steps": 50000
                  "npt_scheme": {1000:50, 316:50, 100:50, 40:120, 16:160, 7:200, 3:225, 1.2:200, 0.5:250, 0.2:225, 0:500}, 
                  "relax_md_tot_time_ns": 30, 
                  "dt": 0.002, "snapshot_interval":25000, 
                  "start_rep": 0, "replicate": 1, "md_snapshot_idx": [0, 599, 1999], "md_tot_time_ns": 100, 
                  "snapshot_begin_at_ps": 0, "delete_old_xtc": True, 
                  "n_threads": 24, "gpu_id":0, 
                  "plumed": "plumed.dat", # or ""
                  "allow_parallel_cpu_for_multiple_sample": False}
    # To run actual simulations, uncomment the line below:
    run_full_simulation(task=task, **components)
