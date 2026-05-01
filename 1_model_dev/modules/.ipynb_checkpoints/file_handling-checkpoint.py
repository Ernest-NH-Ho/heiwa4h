import os
import subprocess
import shutil
from pathlib import Path

def ensure_dirs(params):
    # Create directories if they don't exist
    for name, directory in vars(params.DIRECTORIES).items():
        if name.count("__") == 0:
            os.makedirs(directory, exist_ok=True)
        
def compress(filepath, logger, num_cpu=8):
    """Compress file with parallel gzip (replaces original with .gz)"""
    if os.path.exists(filepath) and not os.path.exists(f"{filepath}.gz"):
        logger.record(f"Compressing {filepath}")
        cmd = f"pigz -p {num_cpu} {filepath}"
        subprocess.run(cmd, shell=True, check=True)
        if os.path.exists(filepath):
            if os.path.isfile(filepath):
                os.remove(filepath)
            else:
                shutil.rmtree(filepath)
        # Now you have filepath.gz, and filepath is gone

def decompress(filepath, logger, num_cpu=8):
    """Decompress file with parallel gzip (replaces .gz with original)"""
    if not os.path.exists(filepath) and os.path.exists(f"{filepath}.gz"):
        logger.record(f"Decompressing {filepath}")
        cmd = f"pigz -d -p {num_cpu} {filepath}.gz"
        subprocess.run(cmd, shell=True, check=True)
        # Now you have filepath, and filepath.gz is gone
        compressed_path = f"{filepath}.gz"
        if os.path.exists(compressed_path):
            os.remove(compressed_path)
            logger.record(f"Removing {compressed_path}")


class FolderTransferer:
    def __init__(self, params):
        self.backup_address = params.BACKUP_ADDRESS
        self.num_cpu = params.NUM_WORKERS
        self.block_factor = params.BLOCK_FACTOR
        self.main_ssd_path = params.MAIN_SSD_ADDRESS
        self.backup_ssd_path = params.BACKUP_SSD_ADDRESS
        
    def _backup(self, params):
        """Backup a folder to backup SSD and remove from main SSD"""
        basename = os.path.basename(params.DIRECTORIES.MODEL_DIR)
        backup_path = f"{self.backup_address}/{basename}.tar.gz"
        
        # Create backup directory if it doesn't exist
        os.makedirs(self.backup_address, exist_ok=True)
        
        # Backup using tar and pigz
        cmd = (
            f"tar --blocking-factor={self.block_factor} -cf - "
            f"{basename} | pigz -p {self.num_cpu} > {backup_path}"
        )
        
        try:
            subprocess.run(cmd, shell=True, cwd=params.DIRECTORIES.PROJ_DIR, check=True, capture_output=True, text=True)
            # Clean up original folder after successful backup
            shutil.rmtree(params.DIRECTORIES.MODEL_DIR)
            print(f"Backed up and removed: {params.DIRECTORIES.MODEL_DIR}")
        except subprocess.CalledProcessError as e:
            print(f"Backup failed for {params.DIRECTORIES.MODEL_DIR}: {e.stderr}")
            raise
    
    def _retrieve(self, params):
        """Retrieve a folder from backup SSD if it doesn't exist on main SSD"""
        if not os.path.exists(params.DIRECTORIES.MODEL_DIR):
            basename = os.path.basename(params.MODEL_DIR)
            backup_path = f"{self.backup_address}/{basename}.tar.gz"
            
            if not os.path.exists(backup_path):
                raise FileNotFoundError(f"Backup not found: {backup_path}")
            
            # Extract to parent directory to avoid nesting
            # parent_dir = os.path.dirname(params.DIRECTORIES.MODEL_DIR)
            os.makedirs(params.DIRECTORIES.MODEL_DIR, exist_ok=True)
            
            strip_count = len(Path(backup_path).parts) - 1 # --strip-components={strip_count}
            cmd = (
                f"tar --use-compress-program='pigz -d -p {self.num_cpu}' "
                f"--blocking-factor={self.block_factor} -xf {basename}.tar.gz -C {params.DIRECTORIES.PROJ_DIR}"
            )
            # os.makedirs(params.DIRECTORIES.MODEL_DIR, exist_ok=True)
            
            try:
                subprocess.run(cmd, shell=True, cwd=self.backup_address, check=True, capture_output=True, text=True)
                # Clean up backup after successful retrieval
                os.remove(backup_path)
                print(f"Retrieved and removed backup: {params.DIRECTORIES.MODEL_DIR}")
            except subprocess.CalledProcessError as e:
                print(f"Retrieval failed for {params.DIRECTORIES.MODEL_DIR}: {e.stderr}")
                raise
    
    def _monitor_capacity(self):
        """Monitor SSD capacity and move folders if needed."""
        # Get disk usage
        main_stat = shutil.disk_usage(self.main_ssd_path)
        backup_stat = shutil.disk_usage(self.backup_ssd_path)
        
        # Calculate percentage used
        main_free_percent = (main_stat.free / main_stat.total) * 100
        backup_free_percent = (backup_stat.free / backup_stat.total) * 100
        
        threshold = 0.03  # Consider "full" at 97%
        
        # Both full - quit
        if main_free_percent < threshold and backup_free_percent < threshold:
            print(f"ERROR: Both SSDs are full!")
            print(f"Main SSD: {main_usage_percent:.1f}% used")
            print(f"Backup SSD: {backup_usage_percent:.1f}% used")
            quit()
        
        # Main SSD full - move 20 most recently modified folders to backup
        if main_free_percent < threshold:
            print(f"Main SSD is {main_usage_percent:.1f}% full. Moving folders to backup...")
            self._move_largest_folders(source="main", count=4)
        
        # Backup SSD full - move 20 most recently modified folders to main
        if backup_free_percent < threshold:
            print(f"Backup SSD is {backup_usage_percent:.1f}% full. Moving folders to main...")
            self._move_largest_folders(source="backup", count=4)
    
    def _move_largest_folders(self, source, count=20):
        """
        Find and move the most recently modified folders.
        
        Args:
            source: "main" or "backup" - indicates which direction to move
            count: Number of folders to move (default 20)
        """
        # Determine search path based on source
        search_path = self.backup_address
        
        # Get most recently modified items using ls -t
        try:
            cmd = f"cd {search_path} && ls -t | head -n {count}"
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            items = result.stdout.strip().split('\n')
        except subprocess.CalledProcessError as e:
            print(f"Failed to list folders in {search_path}: {e.stderr}")
            return
        
        if not items or items == ['']:
            print(f"No items found in {search_path}")
            return
        
        # Filter to only directories
        folders_to_move = []
        items = [os.path.join(search_path, item) for item in items]
        folders_to_move = [item_path for item_path in items if os.path.isdir(item_path)]
        
        if not folders_to_move:
            print(f"No folders found to move from {search_path}")
            return
        
        print(f"Moving {len(folders_to_move)} most recently modified folders...")
        
        for folder in folders_to_move:
            basename = os.path.basename(folder)
            print(f"  Moving {basename}...")
            
            try:
                if source == "main":
                    # Moving from main to backup - compress and move
                    backup_path = f"{self.backup_address}/{basename}.tar.gz"
                    
                    cmd = (
                        f"tar --blocking-factor={self.block_factor} -cf - "
                        f"{folder} | pigz -p {self.num_cpu} > {backup_path}"
                    )
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    shutil.rmtree(folder)
                    print(f"  Moved {basename} to backup")
                    
                else:
                    # Moving from backup to main - compress, extract, cleanup
                    backup_tar = f"{folder}.tar.gz"
                    
                    # Compress
                    cmd = (
                        f"tar --blocking-factor={self.block_factor} -cf - "
                        f"{folder} | pigz -p {self.num_cpu} > {backup_tar}"
                    )
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    
                    # Extract to main
                    parent_dir = os.path.dirname(folder)
                    cmd = (
                        f"tar --use-compress-program='pigz -d -p {self.num_cpu}' "
                        f"--blocking-factor={self.block_factor} -xf {backup_tar} -C {parent_dir}"
                    )
                    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
                    
                    # Cleanup
                    shutil.rmtree(folder)
                    os.remove(backup_tar)
                    print(f"  Moved {basename} to main")
                    
            except Exception as e:
                print(f"  Failed to move {basename}: {e}")
                continue
                