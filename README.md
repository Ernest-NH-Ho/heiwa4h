## Project intro
Hybrid Explicit-Implicit Water Attention (HEIWA) adds aqueous solvent encoding the protein language model ESM3 (https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1), resulting in improved forcefield understanding and higher stability in generated proteins. 

## Purpose of repo
This is three-in-one repo that provide the my source code for
1. HEIWA model training using a proxy thermophilic protein classification task (```1_model_dev/```)
2. ESM3 protein generation with/without HEIWA (```2_esm3_with_heiwa/```)
3. Source code and data for molecular dynamic simulations (```3_mol_dyn/```)

## Conda Environment
```bash
conda create -n heiwa4h python==3.12.2 biopython==1.84 matplotlib==3.9.2 numpy==1.26.4 modeller==10.8 pandas==2.2.3 scikit-learn==1.8.0 scipy==1.14.1 umap-learn==0.5.12
conda activate heiwa4h
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
conda install tqdm==4.67.1 esm==3.2.0
```

-------------------------------------------------------------------------
## Instructions
Please consult the ```README.md``` inside each folder.

## Citations
tbc