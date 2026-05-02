## Purpose
Provide the source code for HEIWA model training, deployment, with training and inference data provided.

## Rationale
The training script was designed to be adaptible to various supervised learning projects, model architectures, learning rates schedules, and hyperparamter combinations.

## Prepare your data
- Save all your data as pt file inside ```data/DATA_FOLDER/```. The data in this study in "data/HEIWA3P".
- Inside ```data/HEIWA3P/```, W0 tensors (```W0_train.pt```, ```W0_valid.pt```, and ```W0_test.pt```) encode the weight for each sample, which accounts for the composition bias of your dataset.
- If the sum of W0 of all positive and negative samples differs significantly, you should define Wp (weight for positive samples) and Wn (weight for negative samples) for each dataset split to balance the binary cross entropy loss (see ```modules/functions/loss.py```).
- M tensors contains boolean mask for padded positions. L tensors contains the sequence length for each sample in X.

## For the regeneration of X tensors
Currently, only X data for inference were provided as the ```X_train.pt```, ```X_valid.ot```, and ```X_test.pt``` are each over 4 Gb after compression. 
Users/reviewers can reproduce our results by running ```inference_run.py``` directly.
But to reproduce our training results, we provide the original ```2_esm3_with_heiwa/input_csv/uniprotkb_length_40_TO_1024_AND_reviewe_2026_03_07_xDupe_sf5_15_split.csv``` and all source dssp files inside ```2_esm3_with_heiwa/original_dataset_dssp/```. Please follow the instructions in ```2_esm3_with_heiwa/README.md``` to regenerate all embeddings to make all X_tensors via concatenation. The running command should be
```bash
python generate_proteins.py -heiwa 0
```

The X tensors can be synthesized by concatenation of all embeddings using right padding into one tensor. The size of tensor would be **n * max_seq_len * 1536** where n is the size of the dataset ```uniprotkb_length_40_TO_1024_AND_reviewe_2026_03_07_xDupe_sf5_15_split.csv```.

- ***Note 1***: max_seq_len is 1024, 1017, and 1021 for ```X_train.pt```, ```X_valid.pt```, and ```X_test.pt```, respectively.
- ***Note 2***: remember to exclude the CLS and EOS tokens from each embedding [size = **(seq_len+2)** x 1536} by slicing ```tensor[1:-1, ...]```.

## Prepare your script

This script is designed to be easily manipulated by only modifying three python scripts ```modules/model/model.py```, ```modules/train_control/parameters.py``` and ```modules/train_control/hp_tuning.py```, plus **two csv files** in your data folder.

1. Customizing model architecture
- Define your model architecture under ```class MyModel``` in ```modules/model/model.py```. Different parts can be defined and named under the ```self.parts```. Line 144-150 controls the loading of pre-trained parameters for each part.
- Create ```grad_schedule.csv``` and ```lr_schedule.csv``` in the same folder of your dataset. All column names, except for ***epoch*** should be identical to the names of your model parts. In ```grad_schedule.csv```, all values are 1 or 0, which switch on/off the gradient of the respective part. Similarly, ```lr_schedule.csv``` controls the learning rate for each model part. You do not need to define optimizers for the model. The script will define it for you.

2. Customizing hyperparameters
- All hyperparameters are stored and can be manually modified under ```class Parameters``` in ```modules/train_control/parameters.py```. You probably like to name the destination of model save at ```self.MODEL_FOLDER_NAME``` (project folder) and ```self.DIRECTORIES.MODEL_DIR``` (subfolder inside project folder, useful for hp_tuning). As such, all your checkpoints, logs would be saved at ```models/MODEL_FOLDER_NAME/MODEL_DIR/```.
- According to your hardware, you may set ```self.BATCH_SIZE``` and ```self.ACCUM_STEP``` (effective batch size = self.BATCH_SIZE * self.ACCUM_STEP).
- Number of CPU can be set at ```self.NUM_WORKERS```
- Name the location of dataset at ```self.DATA_FOLDER``` if your tensors are saved at ```data/DATA_FOLDER/```.
- For multiple hp tuning, open ```modules/train_control/hp_tuning.py``` and set ```self.hp_to_try``` in ```class HyperparameterTuner```. The name of the keys would be added to ```class Parameters``` as add-on attributes.

## Run
```bash
python train_run.py -head 4
python inference_run.py -save_intermediate_layer 1 -head 4
```

The expected outputs from training and inference runs are included in ```models/HEIWA_SelfGate_rot1_angAtten1_exhd4_Imp_newSASAgateSoft/esm3_SelfGate_bs64_dr1_3_v192_v_compress0.8_Imp_hd4/```
