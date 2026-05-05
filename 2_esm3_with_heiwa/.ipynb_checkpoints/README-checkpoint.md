## Purpose

Generate de novo proteins using ESM3 with/without HEIWA add-in.

## Installation
### Step 1: Install DSSP

Download and install DSSP from the official releases:
- https://github.com/cmbi/dssp/releases

### Step 2: Install ESM3
1. Obtain the ESM3 model using the official public license from HuggingFace:
- Visit: https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1
- Follow the licensing requirements and download instructions
2. Save the model as ```esm_model/esm3.pth```
3. Replace the following files using the python scripts provided inside ```esm_model/```
- ```<conda_env>/esm/utils/generation.py```
- ```<conda_env>/esm/models/esm3.py```

## Prepare HEIWA model weights

Transfer the model weights from ```1_model_dev``` to this directory.
```bash
for part in ExposedResiduePrediction HEIWA_Hybridizer HEIWA_Imp TIM; do
    cp ../1_model_dev/HEIWA_SelfGate_rot1_angAtten1_exhd4_Imp_newSASAgateSoft/esm3_SelfGate_bs64_dr1_3_v192_v_compress0.8_Imp_hd4/${part}_epoch_12_model.pth.gz heiwa/${part}_epoch_final_model.pth.gz
done
```

## Prepare your data

### DSSP File Preparation

Place input DSSP files in the `data/` directory

### CSV File Preparation
Place input CSV files in the `input_csv/` directory

The CSV file must contain the following fields:

| Field | Description |
|-------|-------------|
| `Entry` | Sequence ID |
| `Sequence_final` | Valid amino acid sequence string for ESM3 input |
| `Length_final` | Length of the sequence in `Sequence_final` |
| `DSSP_start` | Position of the first residue of `Sequence_final` in PDB (0 if sequences are identical, must be ≥ 0) |
| `Act_bind_motif_sites` | Critical residues in `Sequence_final` (first residue has index 1) |
| `Filename` | File name of your protein dssp file, excluding the suffix **.dssp** |

**Important Notes:**
- Make sure each sequence ID is unique in the csv
- Do not include sequences over 1,650 amino acids for ESM3 input if you are running on a single RTX 3090ti.
- Make sure CSV sequences are identical to PDB sequences after accounting for the shifts in ```DSSP_start```

## Prepare your script
For most users, most, if not all, configuration can be done by editing ```modules/config.py```
You may like to 
- Set the number of CPU at ```self.NUM_CPU```
- Set the location of your DSSP file at ```self.DSSP_DIR```
- Set the location of your csv file at ```self.CSV_PATH```
- At ```self.ESM3_TRACKS```, disable the last 8-iter 'sequence' track to regenerate X tensors, enable it to generate de novo sequences.
- Set the output folder name ```self.DATASET_NAME```. As such your embedding and protein objects would be saved in ```esm3_embeddings/DATASET_NAME``` and ```esm3_proteins/DATASET_NAME```.

### Run
To deploy ESM3 for protein generation with heiwa
```bash
python generate_proteins.py -heiwa 1
```

To deploy ESM3 for protein generation without heiwa
```bash
python generate_proteins.py -heiwa 0
```