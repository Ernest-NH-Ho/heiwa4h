## Project intro
Hybrid Explicit-Implicit Water Attention (HEIWA) is a correction to the forcefield distortion of protein language model ESM3 (https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1). 
Time-averaged protein-solvent interactions were encoded as static embedding that can be used to modify the SE(3)-embedding of ESM3 via an all-to-all attention-based message passing.

## Purpose of repo
This is three-in-one repo that provide the my source code for
1. HEIWA model training using a proxy thermophilic protein classification task
2. ESM3 protein generation with HEIWA
3. Molecular dynamic simulations

## Conda Environment
```bash
conda env create -n heiwa4h python==3.12.2 biopython==1.84 matplotlib==3.9.2 numpy==1.26.4 modeller==10.8 pandas==2.2.3 scikit-learn==1.8.0 scipy==1.14.1 umap-learn==0.5.12
conda activate heiwa4h
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install tqdm==4.67.1 esm==3.2.0
```

-------------------------------------------------------------------------
## Instructions
Please consult the ```README.md``` inside each folder.

## Citations
tbc