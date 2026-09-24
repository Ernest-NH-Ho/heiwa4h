## Project Introduction
Hybrid Explicit-Implicit Water Attention (HEIWA) adds aqueous solvent encoding the protein language model ESM3 (https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1), resulting in improved forcefield understanding and higher stability in generated proteins. 

## For Easy Implementation of HEIWA
1. Conda environment
```bash
python write_env_yaml.py --name <name_of_env> --torch <my_version> --cuda <depends_on_your_gpu>
conda env create -f environment.yaml
```

WARNING: heiwa does modify the esm3 script. We recommend a separate environment.

2. Python implementation
```python
import heiwa
import torch
from esm.models.esm3 import ESM3

# ===== Hardware setting =====
hw_setting = heiwa.Hardware_Setting(
    cpu=16, 
    batch_size=16, 
    gpu_rs=12345, # only triggered by gen_embed, de_novo, or predict functions
    load_device = "cpu", 
    run_device = "cuda",
)


# ===== Load models  =====
esm3_model = ESM3.from_pretrained(
    model_name="esm3_sm_open_v1", 
    device=torch.device("cuda")
)

heiwa_model, heiwa_orthogonalizer = heiwa.load_solv_encoder(hw_setting) # 

thermo_model = heiwa.load_thermo(hw_setting)


# ===== Generation commands (Only accept 1 file per run) =====
heiwa.gen_embed(esm3_model=esm3_model, hw_setting=hw_setting, inFile="a_single_protein.pdb", outPath="./a_single_protein.pt")
heiwa.de_novo(
    hw_setting=hw_setting, 
    esm3_model=esm3_model,
    heiwa_model=heiwa_model, 
    heiwa_orthogonalizer=heiwa_orthogonalizer, 
    cleave=0, # for signal peptide
    seq_prompt=seq_prompt, # Option 1 for prompt (choose one option only)
    crit_resi = (45, 59, 61, 64, 66, 69, 93, 187), # Option 2 for prompt
    idx_style=1, # Option 2 only (0 or 1)
    inFile="a_single_protein.cif", 
    outPath="a_single_protein", 
)

# ===== Predict thermophilicity of a protein ====
### Load 1 tensor or 1 path
heiwa.predict(
    heiwa_model=heiwa_model, 
    heiwa_orthogonalizer=heiwa_orthogonalizer, 
    thermo_model=thermo_model, 
    x=tensor, # Option 1: directly load a tensor --> exclude_cls & exclude_eos must be None
    in_path,  # Option 2: give the directory
    cleave,   # Option 2
    out_folder=".", 
    hw_setting=hw_setting, 
    exclude_cls=False, # Option 2
    exclude_eos=False, # Option 2
)

### Load multiple tensors or multiple path (Choose 1 option)
heiwa.predict(
    heiwa_model=heiwa_model, 
    heiwa_orthogonalizer=heiwa_orthogonalizer, 
    thermo_model=thermo_model, 
    x=tensors,         # Option 1: tensors is a list of tensor --> exclude_cls & exclude_eos must be None
    in_path=[
        "embedding_1.pt", # Option 2: a list of input embedding file
        "embedding_2.pt", 
        "embedding_3.pt", 
        "embedding_4.pt", 
    ], 
    cleave=[0, 0, 0, 0], # Option 2
    out_folder=".", 
    hw_setting=hw_setting, 
    exclude_cls=[True, True, True, True], # Option 2
    exclude_eos=[True, True, True, True], # Option 2
)
```

Notes:
- Input files can be pdb, cif, dssp

## For reviewers and whoever pursuing manual inspection
This is three-in-one repo that provide the my source code for
1. HEIWA model training using a proxy thermophilic protein classification task (```1_model_dev/```)
2. ESM3 protein generation with/without HEIWA (```2_esm3_with_heiwa/```)
3. Source code and data for molecular dynamic simulations (```3_mol_dyn/```)

### Conda Environment
```bash
conda create -n heiwa4h python==3.12.2 biopython==1.84 matplotlib==3.9.2 numpy==1.26.4 modeller==10.8 pandas==2.2.3 scikit-learn==1.8.0 scipy==1.14.1 umap-learn==0.5.12
conda activate heiwa4h
conda install pytorch==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install tqdm==4.67.1 esm==3.2.0
```

-------------------------------------------------------------------------
## Other Instructions
Please consult the ```README.md``` inside each folder.

## Version History & Documentation
https://pypi.org/project/heiwa/

## Citations
tbc
