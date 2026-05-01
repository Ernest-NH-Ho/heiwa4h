## Purpose
Provide the source code for HEIWA model training and deployment and inference data.

## Rationale
The training script was designed to be adaptible to various supervised learning projects, with flexible model architecture, learning rates schedules, and hyperparamter combinations.

## Prepare your data
- Save all your data as pt file inside ```data/DATA_FOLDER/```. A demo as been done where DATA_FOLDER is "HEIWA3P".
- W0 encodes the weight for each sample, which accounts for the composition bias of your dataset.
- If the sum of W0 of all positive and negative samples differs significant, you should define Wp (weight for positive samples) and Wn (weight for negative samples) for balancing the binary cross entropy loss (see ```modules/functions/loss.py```).
- M tensors contains boolean mask for padded positions. L tensors contains the sequence length for each sample in X.

## Prepare your script

This script is designed to be easily manipulated by only modifying three python scripts ```model.py```, ```parameters.py``` and ```hp_tuning.py```, plus two csv files in your data folder.

1. Customize model architecture
- Define your model architecture under ```class MyModel``` in ```modules/model/model.py```. Different parts can be defined and named under the ```self.parts```. Line 144-150 controls the loading of pre-trained model weights for each part.
- Create ```grad_schedule.csv``` and ```lr_schedule.csv``` in the same folder of your project dataset. All column name, except for ***epoch*** should be identical to the names of your model parts. In ```grad_schedule.csv```, all values are 1 or 0, which switch on/off the gradient of the respective part.
- Similarly, ```lr_schedule.csv``` controls the learning rate for each model part. You do not need to define optimizers for the model. The script will define it for you.
2. Customize hyperparameters
- All hyperparameters are stored and can be manually modified under ```class Parameters``` in ```modules/train_control/parameters.py```. You probably like to name the destination of model save at line 87 at ```self.MODEL_FOLDER_NAME``` (project folder) and line 46 at ```self.DIRECTORIES.MODEL_DIR``` (subfolder inside project folder, useful for hp_tuning). As such, all your checkpoints, logs would be saved at ```models/MODEL_FOLDER_NAME/MODEL_DIR/```.
- According to your hardware, you may set ```self.BATCH_SIZE``` and ```self.ACCUM_STEP``` (final batch size = self.BATCH_SIZE * self.ACCUM_STEP).
- Number of CPU can be set at ```self.NUM_WORKERS
- Name the location of dataset at ```self.DATA_FOLDER``` if your tensors are saved at ```data/DATA_FOLDER/```.
- For multiple hp tuning, open ```modules/train_control/hp_tuning.py``` and set ```self.hp_to_try``` in ```class HyperparameterTuner```. The name of the keys would be added to ```class Parameters``` as add-on attributes.

## Run
```bash
python train_run.py -head 4
python inference_run.py -save_intermediate_layer 1 -head 4
```

The expected outputs from training and inference runs are included in ```models/HEIWA_SelfGate_rot1_angAtten1_exhd4_Imp_newSASAgateSoft/esm3_SelfGate_bs64_dr1_3_v192_v_compress0.8_Imp_hd4/```

Currently, only X data for inference were provided as the X tensors for train, validation, and test are each over 4 Gb after compression. Instead, we provided the original ```uniprotkb_length_40_TO_1024_AND_reviewe_2026_03_07_xDupe_sf5_15_split.csv```. Obtaining the AlphaFold
