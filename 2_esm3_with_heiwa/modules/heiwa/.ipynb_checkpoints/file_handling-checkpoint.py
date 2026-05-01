import os
import subprocess
import shutil
from pathlib import Path

def ensure_dirs(params):
    # Create directories if they don't exist
    for name, directory in vars(params.DIRECTORIES).items():
        if name.count("__") == 0:
            os.makedirs(directory, exist_ok=True)
        
def compress(filepath, num_cpu=8):
    """Compress file with parallel gzip (replaces original with .gz)"""
    if os.path.exists(filepath) and not os.path.exists(f"{filepath}.gz"):
        cmd = f"pigz -p {num_cpu} {filepath}"
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(filepath):
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath)
        # Now you have filepath.gz, and filepath is gone

def decompress(filepath, num_cpu=8):
    """Decompress file with parallel gzip (replaces .gz with original)"""
    if not os.path.exists(filepath) and os.path.exists(f"{filepath}.gz"):
        cmd = f"pigz -d -p {num_cpu} {filepath}.gz"
        subprocess.run(cmd, shell=True, check=True)
        # Now you have filepath, and filepath.gz is gone
        compressed_path = f"{filepath}.gz"
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
