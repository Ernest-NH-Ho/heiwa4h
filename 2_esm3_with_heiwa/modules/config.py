import hashlib
import os

def mix_seeds(*seeds):
    seed_str = "_".join(map(str, seeds))
    hashed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16)
    return hashed % (2**63)  # PyTorch expects 63-bit seeds
all_seeds = [
    12345, 5555555, 777777777, 15657, 
    456789, 6444445, 1025555, 1536789,  
    9999999, 7890123, 1111111111, 33333333333,  
    1289154545, 512456447, 81923157, 25631313
]
all_seeds = [i+2 for i in all_seeds]
random_seed = mix_seeds(*all_seeds) #  

class DataConfig:
    def __init__(self):
        # Device
        self.NUM_CPU = 24
        self.DEVICE = 'cuda'

        # Directories
        self.DATASET_NAME = '30repr_202604_192_224_heiwa1_softSASA' ## Please name your dataset
        self.BASE_DIR = "."
        self.DSSP_DIR = f"/home/user/Databases/AlphaFold_ACT_BIND_dssp"
        self.PDB_DIR = f"/home/user/Databases/AlphaFold_ACT_BIND"
        self.DATA_DIR = f"{self.BASE_DIR}/data/{self.DATASET_NAME}"

        # CSV FILE INPUT
        self.MODE = "cleavage" # or "naive" <-- naive mode is obsolete
        self.CSV_PATH = f"input_csv/uniprotkb_length_40_TO_1024_AND_reviewe_2026_03_07_xDupe_sf226_192_224_sample30.csv"
        
        # DSSP_config
        self.DSSP_OUTPUT = ["sequence"]
        
        # esm3_config
        self.ESM3_PATH = f"{self.BASE_DIR}/esm3_model/esm3.pth"
        self.ESM3_EMB_DIR = f"{self.BASE_DIR}/esm3_embeddings/{self.DATASET_NAME}"
        self.ESM3_PROT_DIR = f"{self.BASE_DIR}/esm3_proteins/{self.DATASET_NAME}"
        self.MAX_LEN = 1280  # Depends on your GPU memory capacity
        self.ESM3_TRACKS = [{'structure':16},
                            {'secondary_structure':8}, 
                            {'sasa':4}, 
                            {'structure':8}, 
                            {'sequence':8}]
        self.ESM3_TEMP = 0.7

        # PCA
        self.PCA_DIR = f"{self.BASE_DIR}/pca/structure,secondary_structure,sasa_structure_wifMulti_l-1"
       
        # Make all directories
        self._ensure_dirs()
        
    def _ensure_dirs(self):
        for folder in (self.DSSP_DIR, self.PDB_DIR, self.DATA_DIR, 
                       self.ESM3_EMB_DIR, self.ESM3_PROT_DIR, 
                       self.PCA_DIR, f"{self.ESM3_EMB_DIR}_pca"):
            if not os.path.exists(folder):
                os.makedirs(folder)
                
