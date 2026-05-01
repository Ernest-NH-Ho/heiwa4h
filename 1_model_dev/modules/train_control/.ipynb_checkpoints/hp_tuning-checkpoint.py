import pandas as pd
import itertools
import math
import os
import re
import numpy as np
from modules.train_control.parameters import ConstantParameters, Parameters

class HyperparameterTuner_Instruction:
    def __init__(self):
        self.numb_cycle = 0
        self.start_epoch = 0
        if self.numb_cycle == 0:
            self.extra_epoch_start = 4
        else:
            self.extra_epoch_start = 0
        self.epochs_by_steps = 3 # it shall increase the epochs
        self.end_epoch = self.start_epoch + self.epochs_by_steps + self.extra_epoch_start

    def receive_combo_specific_instruction(self, status_instructions, hp_instructions, dtype, params):
        self.status_instructions = status_instructions
        self.hp_instructions = hp_instructions
        self.dtype = dtype
        self.end_epoch = params.MAX_EPOCHS if self.tot_combo_num == 1 else self.end_epoch

    def __copy__(self):
        new_copy = HyperparameterTuner_Instruction()
        for name, value in self.__dict__.items():
            setattr(new_copy, name, value)
        return new_copy

    def harmonize_instructions(self, tasks, logger):
        if "train" in tasks:
            for col in ['cycle_num', "earlystop_counter", "best_epoch", "last_epoch"]:
                self.status_instructions[col] = int(self.status_instructions[col])
            self.start_epoch = min(self.start_epoch, int(self.status_instructions["last_epoch"])+1)
            logger.record(f"Start epoch is {self.start_epoch} after hp_instruction harmonization.")
            # print(self.start_epoch)
            # print(self.status_instructions["last_epoch"])
            # print(self.end_epoch)
        
    def __str__(self):
        """ There is \n automatically after printing """
        cout = f""
        if isinstance(self.status_instructions, pd.Series):
            for name, value in self.status_instructions.items():
                if name not in ['best_test_metric', 'cycle_num', 'ready_to_stop']:
                    cout += f"{name} = {value}  "
        for name, value in self.hp_instructions.items():
            if name not in ['best_test_metric', 'cycle_num', 'ready_to_stop']:
                cout += f"{name} = {value}  "
        if isinstance(self.status_instructions, pd.Series):
            cout += f"start_epoch = {min(self.start_epoch, self.status_instructions["last_epoch"])}  "
        cout += f"end_epoch = {self.end_epoch}"
        return cout


class HyperparameterTuning_Report:
    """ After each combo, the train_run should report their metrics and earlystop performance to the hp_tuner """
    def __init__(self, logbook, earlystop_report=None, updated_dropout=None):
        self.logbook = logbook
        self.earlystop_report = earlystop_report
        self.updated_dropout = updated_dropout

class HyperparameterTuner:
    """ It should not receive params as input """
    def __init__(self, logger, params:ConstantParameters, tasks=["train", "valid", "test"]):
        self.tasks = tasks
        self.instructions = HyperparameterTuner_Instruction()
        self.pc_to_keep = 0.6
        logger.record(f"*** Hp_tuner initialize the following parameters ***")
        logger.record(f"tasks = {tasks}")
        logger.record(f"precentage of filter = {self.pc_to_keep}")
        
        # Manually enter the hp to tune
        self.hp_to_try = {}
        
        self.hp_to_try["value"] = {"heiwa_imp_v_dim": [192],
                                   "hyb_v_compress": [0.8], 
                          }
        self.hp_to_try["dtype"] = {"heiwa_imp_v_dim": int, 
                                   "hyb_v_compress": float,
                          }

        for k, v in self.hp_to_try.items():
            logger.record(f"{k}: {v}")
            
        # =========== Fresh start vs Middle start ====================
        file_list = [file for file in os.listdir(f"{params.DIRECTORIES.PROJ_DIR}") if file.startswith(f"hp_tuning_combo") and file.endswith(".csv")]

        # Fresh start for self.combinations
        self.combinations = list(itertools.product(*list(self.hp_to_try["value"].values())))
        self.combinations = pd.DataFrame(self.combinations, columns=list(self.hp_to_try["value"].keys()))
        if "train" in self.tasks:
            self.combinations["earlystop_counter"] = 0
            self.combinations["best_epoch"] = 0
            self.combinations["last_epoch"] = 0
            self.combinations["cycle_num"] = 0
            self.combinations["least_overfit"]= np.Inf
            logger.record(f"Hp_tuner initialize the following training parameters")
            logger.record(f"least_overfit: {self.combinations['least_overfit']}")
            logger.record(f"best_epoch: {self.combinations['best_epoch']}")
            logger.record(f"last_epoch: {self.combinations['last_epoch']}")
        self.instructions.tot_combo_num = len(self.combinations)
        
    def find_latest_record(self, params, logger): 
        # Find the latest hp_tuning record, if any
        logger.record(f"Search for the latest hp_tuning record csv")
        start_cycle = -1
        for file in os.listdir(f"{params.DIRECTORIES.PROJ_DIR}"):
            if file.startswith("hp_tuning_combo_"):
                start_cycle = max(start_cycle, int(re.findall(r"hp_tuning_combo_([0-9]+)", file)[0]))
        logger.record(f"Will start at cycle {start_cycle}")
        # If there is a record?
        if start_cycle >= 0:
            self.combinations = pd.read_csv(f"{params.DIRECTORIES.PROJ_DIR}/hp_tuning_combo_{start_cycle}.csv", header=0)
            logger.record(f"hp_tuning_combo_{start_cycle}.csv loaded")
            
            for _ in range(start_cycle+1):
                self.next_cycle(params=params, logger=logger)

            # Supposed tot_combo_num at start = min(max_intended_combo, numb_of_not_ready_2_stop)
            self.instructions.tot_combo_num = min(self.instructions.tot_combo_num, 
                                                  len(self.combinations[self.combinations["ready_to_stop"]==0])) + len(self.combinations[self.combinations["best_test_metric"].isna()])

        return start_cycle + 1

    def is_exception(self):
        return False
        
    def give_instructions(self, i, params, logger):
        dropout_cols = [col for col in self.combinations.columns if "dropout" in col]
        logger.record(f"dropout field: {dropout_cols}")
        if "train" in self.tasks:
            status_instructions = self.combinations.loc[i, \
            ["earlystop_counter", "cycle_num", "best_epoch", "last_epoch", "least_overfit"]+dropout_cols]
        else:
            status_instructions = None
            
        logger.record(f"status instrations field: {status_instructions}")
        self.instructions.receive_combo_specific_instruction(
            status_instructions=status_instructions, 
            hp_instructions=self.combinations.loc[i, \
            list(self.hp_to_try["value"].keys())], 
            dtype=self.hp_to_try["dtype"], 
            params=params
        )
        return self.instructions

    def update_instructions(self, logger):
        """ Move instruction one step up """
        self.instructions.numb_cycle += 1
        logger.record(f"Cycle number is now {self.instructions.numb_cycle}")
        self.instructions.start_epoch += self.instructions.epochs_by_steps + self.instructions.extra_epoch_start
        logger.record(f"start epoch updated to: {self.instructions.start_epoch}")
        self.instructions.extra_epoch_start = 0
        if self.instructions.numb_cycle %2 == 0:
            self.instructions.epochs_by_steps += 1
            logger.record(f"epochs increment: {self.instructions.epochs_by_steps}")
        self.instructions.end_epoch = self.instructions.start_epoch + self.instructions.epochs_by_steps + self.instructions.extra_epoch_start
        logger.record(f"End epoch: {self.instructions.epochs_by_steps}")

    def conclude_and_record_current_cycle(self, params:Parameters):
        # Sort the best combo
        self.combinations = self.combinations.sort_values(
            by=['best_test_metric', 'best_valid_metric', 'best_train_metric'], 
            ascending=params.EARLYSTOP_MODE=='min', 
            kind='mergesort'
        )
        if self.instructions.tot_combo_num > 1 and len(self.combinations["ready_to_stop"].unique()) > 1:
            self.combinations = self.combinations.sort_values(
                by=['ready_to_stop'], 
                ascending=True, 
                kind='mergesort'
            )
        self.combinations = self.combinations.reset_index(drop=True)
        self.combinations.to_csv(
            f"{params.DIRECTORIES.PROJ_DIR}/hp_tuning_combo_{self.instructions.numb_cycle}.csv", 
            index=False
        )
    
    def next_cycle(self, logger, params:Parameters):
        logger.record(f"Prepare to move on the next hp_tuning cycle")
        self.update_instructions(logger=logger)
        # Filter
        self.instructions.tot_combo_num = math.floor(self.instructions.tot_combo_num * self.pc_to_keep)
        logger.record(f"Total combo: {self.instructions.tot_combo_num}")
        
        if self.instructions.tot_combo_num == 1:
            self.instructions.end_epoch = params.MAX_EPOCHS
        elif self.instructions.tot_combo_num == 0:
            self.combinations.sort_values(
                by=["best_test_metric", 'best_valid_metric', 'best_train_metric'], 
                ascending=params.EARLYSTOP_MODE=='min', 
                kind="mergesort"
            )
            self.combinations["BestEpoch_over3"] = (self.combinations["best_epoch"]>3).astype(int)
            self.combinations = self.combinations.sort_values(by=["BestEpoch_over3"], ascending=False, kind="mergesort").reset_index(drop=True)
            self.combinations.to_csv(
                f"{params.DIRECTORIES.PROJ_DIR}/hp_tuning_combo_final.csv", 
                index=False
            )

    def read_combo_metric(self, params, hp_tuning_report, i:int):
        """ i is the index of the hp_tuner object """
        """ At the end of each cycle, read the metric reports """
        """ Assume inference won't touch this """
        logbook = hp_tuning_report.logbook

        # Find the test metric
        if "train" in self.tasks:
            test_idx = logbook.df[(logbook.df["train_valid_test"]=="test") & \
            (logbook.df["epoch"]==hp_tuning_report.earlystop_report.best_epoch)].index.item()
        else:
            test_epoch = logbook.df.loc[(logbook.df["train_valid_test"]=="test"), "epoch"].max()
            test_idx = logbook.df[(logbook.df["train_valid_test"]=="test") & \
            (logbook.df["epoch"]==test_epoch)].index.item()
        
        if params.EARLYSTOP_MODE=='min':
            best_test_metric = logbook.df.loc[test_idx, params.EARLYSTOP_METRICS].max() # .min()
        else:
            best_test_metric = logbook.df.loc[test_idx, params.EARLYSTOP_METRICS].min() # .max() this is a pd.Series
        
        if "train" in self.tasks:
            self.combinations.loc[i, "best_train_metric"] = hp_tuning_report.earlystop_report.best_train_metric
            self.combinations.loc[i, "best_valid_metric"] = hp_tuning_report.earlystop_report.best_valid_metric
            self.combinations.loc[i, "least_overfit"] = hp_tuning_report.earlystop_report.least_overfit
            self.combinations.loc[i, "last_epoch"] = hp_tuning_report.earlystop_report.last_epoch
            self.combinations.loc[i, "ready_to_stop"] = hp_tuning_report.earlystop_report.ready_to_stop
            self.combinations.loc[i, "cycle_num"] += 1
        self.combinations.loc[i, "best_test_metric"] = best_test_metric

        if "train" in self.tasks:
            # Read dropouts for future use
            dropout_cols = [col for col in logbook.df.columns if "dropout" in col]
            # print("HI", hp_tuning_report.updated_dropout, "\n\n\n\n\n\n")
            # quit()
            if hp_tuning_report.updated_dropout == None:
                epoch = logbook.df["epoch"].max()
                
                train_idx = logbook.df[(logbook.df["train_valid_test"]=="train") & (logbook.df["epoch"]==epoch)].index
                
                train_idx = train_idx.item()
                self.combinations.loc[i, dropout_cols] = logbook.df.loc[train_idx, dropout_cols]
            else:
                self.combinations.loc[i, dropout_cols] = pd.Series(hp_tuning_report.updated_dropout)
            
            
            # Report adjusted dropouts for future use
            for name, value in hp_tuning_report.earlystop_report.__dict__.items():
                self.combinations.loc[i, name] = value
            

    def read_combo_metric_from_scratch(self, params, i):
        logbook_df = pd.read_csv(f"{params.DIRECTORIES.MODEL_DIR}/logbook.csv", header=0)
        
        if logbook_df["epoch"].max() < self.instructions.end_epoch-1:
            already_done = False
            print(f"{params.DIRECTORIES.MODEL_DIR} not done")
            return already_done
            
        print(f"{params.DIRECTORIES.MODEL_DIR} already done")
        
        # Find the best metrics for train valid test
        for phase in ["train", "valid", "test"]:
            epoch = logbook_df.loc[(logbook_df["train_valid_test"]==phase) & (logbook_df["epoch"]<=self.instructions.end_epoch-1), "epoch"].max()
            idx = logbook_df.loc[(logbook_df["train_valid_test"]==phase) & (logbook_df["epoch"]==epoch)].index.max()
            
            if params.EARLYSTOP_MODE=='min':
                best_metric = logbook_df.loc[idx, params.EARLYSTOP_METRICS].max()
            else:
                best_metric = logbook_df.loc[idx, params.EARLYSTOP_METRICS].min()
            
            self.combinations.loc[i, f"best_{phase}_metric"] = best_metric
            
            if phase == "test":
                test_epoch = epoch
                test_idx = idx
                
        # Record the epoch and earlystop status
        max_epoch = logbook_df["epoch"].max()
        self.combinations.loc[i, "last_epoch"] = max_epoch
        self.combinations.loc[i, "earlystop_encounter"] = max_epoch - test_epoch
        self.combinations.loc[i, "cycle_num"] += 1
        self.combinations.loc[i, "best_epoch"] = logbook_df.loc[test_idx, "epoch"]
        
        # Read dropouts for future use
        dropout_cols = [col for col in logbook_df.columns if "dropout" in col]
        train_idx = logbook_df[(logbook_df["train_valid_test"]=="train") & (logbook_df["epoch"]==max_epoch)].index
        train_idx = train_idx.item()
        self.combinations.loc[i, dropout_cols] = logbook_df.loc[train_idx, dropout_cols]

        if self.combinations.loc[i, "best_epoch"] > 0:
            self.combinations.loc[i, "patience_index"] = 1
            self.combinations.loc[i, "ready_to_stop"] = int(self.combinations.loc[i, "earlystop_encounter"] > params.PATIENCE)
        else:
            self.combinations.loc[i, "ready_to_stop"] = int(self.combinations.loc[i, "earlystop_encounter"] > params.INIT_PATIENCE)
        
        already_done = True
        return already_done
        
        