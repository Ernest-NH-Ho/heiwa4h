import os
import pandas as pd
from modules.model.model import MyModel
from modules.train_log import Logger, LogBook
from modules.train_control.parameters import ConstantParameters, TunableParameters, Parameters
from modules.train_control.train_schedule import TrainSchedule
from modules.train_control.earlystop import EarlyStopping
from modules.train_control.hp_tuning import HyperparameterTuner_Instruction
from modules.train_control.utils import RandomSeedMix_Generator, import_data

class Components:
    """"""
    def __init__(self):
        pass

    def __add__(self, other:"Components"):
        for name, value in other.__dict__.items():
            setattr(self, name, value)
        return self
        
class ConstantComponents(Components):
    """ Components not affected by parameter tuner """
    def __init__(self, params:ConstantParameters, tasks:list=['train', 'valid', 'test']):
        super().__init__()
        self.tasks = tasks
        self.logger_constant = Logger(f"{params.DIRECTORIES.PROJ_DIR}/record_{params.now_constant}.log")
        self.rs_generator = RandomSeedMix_Generator()
        if "train" in self.tasks:
            self.dataset_train_valid = import_data(train_valid_test=["train", "valid"], params=params, logger=self.logger_constant)
            self.dataset_test = import_data(train_valid_test=["test"], params=params, logger=self.logger_constant)
        elif "test" in self.tasks:
            self.dataset_test = import_data(train_valid_test=["test"], params=params, logger=self.logger_constant)
        elif "inference" in self.tasks:
            self.dataset_inference = import_data(train_valid_test=["inference"], params=params, logger=self.logger_constant)
        # if "train" in tasks:
        #     self.grad_schedule = pd.read_csv(f"{params.DIRECTORIES.DATA_PATH}/grad_schedule.csv", header=0).astype(bool)

class TunableComponents(Components):
    """ Components subject to hyperparameter tuning """
    def __init__(self, params:Parameters, test_epoch:str|int|None=None, 
                 hp_instructions:HyperparameterTuner_Instruction|None=None, 
                 components:ConstantComponents|None = None):
        # |
        super().__init__()
        if components != None:
            self = self + components
        self.logger_tunable = Logger(f"{params.DIRECTORIES.LOG_DIR}/record_{params.now_tunable}.log")
        if "train" in self.tasks:
            self.start_epoch = hp_instructions.start_epoch if hp_instructions != None else 0
            self.test_epoch = test_epoch
            self.initialize_train_schedule(params)
            self.initialize_model(
                params=params, 
                hp_instructions=hp_instructions, 
            )
            self.initialize_logbook(params) # Must be b4 earlystop, it needs logbook
            self.initialize_earlystopping(
                params=params, 
                status_instructions=hp_instructions.status_instructions
            )
        else:
            test_epoch = self.initialize_logbook(params) # Must be b4 earlystop, it needs logbook
            self.initialize_model(
                params=params, 
                hp_instructions=hp_instructions, 
                test_epoch=test_epoch, 
            )

    def initialize_model(self, params:TunableParameters, hp_instructions:HyperparameterTuner_Instruction|None=None, test_epoch=None):
        self.logger_tunable.record("Initializing model...")
        if hp_instructions is not None:
            status_instructions = hp_instructions.status_instructions
        else:
            status_instructions = None
        self.model = MyModel(
            expected_tasks=self.tasks, 
            params=params,
            logger=self.logger_tunable, 
            status_instructions=status_instructions,
        ).to(params.DEVICE).float()
        
        if "train" in self.tasks:
            if self.start_epoch>0:
                self.model._loadpretrain(
                    hp_folder=params.DIRECTORIES.HP_FOLDER, 
                    epoch=self.start_epoch-1, 
                    task='train', 
                    logger=self.logger_tunable, 
                    device=params.DEVICE
                )
        

    def initialize_train_schedule(self, params:TunableParameters):
        # Initialize training schedule
        self.train_schedule = TrainSchedule(max_epochs=params.MAX_EPOCHS, batch_size=params.BATCH_SIZE)
        if self.test_epoch != 'final':
            self.train_schedule.df.to_csv(f"{params.DIRECTORIES.HP_FOLDER}/train_schedule.csv", index=False)
        else:
            self.train_schedule.df = pd.read_csv(f"{params.DIRECTORIES.HP_FOLDER}/train_schedule.csv", header=0)
        self.logger_tunable.record("Training schedule created")
        self.logger_tunable.record(self.train_schedule.get_schedule_summary())

    def initialize_logbook(self, params:TunableParameters):
        self.logbook = LogBook(params)
        if "train" in self.tasks:
            if self.start_epoch>0:
                if os.path.exists(f"{params.DIRECTORIES.MODEL_DIR}/logbook_temp.csv"):
                    file_path = f"{params.DIRECTORIES.MODEL_DIR}/logbook_temp.csv"
                elif os.path.exists(f"{params.DIRECTORIES.MODEL_DIR}/logbook.csv"):
                    file_path = f"{params.DIRECTORIES.MODEL_DIR}/logbook.csv"
                else:
                    for file in os.listdir(f"{params.DIRECTORIES.MODEL_DIR}"):
                        if file.startswith("logbook") and file.endswith(".csv"):
                            file_path = f"{params.DIRECTORIES.MODEL_DIR}/{file}"
                            break
                self.logbook.load_csv(file_path=file_path, merge = True)
            
        else:
            self.logbook.load_csv(file_path=f"{params.DIRECTORIES.MODEL_DIR}/logbook.csv", merge = True)
            self.logbook.df["epoch"] = self.logbook.df["epoch"].astype(int)
            return self.logbook.get_best_epoch()

    def initialize_earlystopping(self, params:TunableParameters, status_instructions:pd.Series|None=None):
        # Initialize early stopping
        self.early_stop = EarlyStopping(
            verbose=True, save=True, params=params, status_instructions=status_instructions, logbook=self.logbook
        )

        if self.start_epoch>0 and status_instructions is not None:
            self.early_stop.read_best_metrics_from_logbook(
                params=params, logbook=self.logbook
            ) # best_epoch=status_instructions["best_epoch"]
            